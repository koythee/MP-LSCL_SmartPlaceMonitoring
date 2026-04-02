import os
import cv2
import json
import logging
import threading
import numpy as np
import tkinter as tk
from mysql.connector import pooling, Error
from GetFrame import Camera
from datetime import datetime
from PIL import Image, ImageTk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Callable
from enum import Enum
from IMVApi import *
import winsound
from pathlib import Path
import csv
import calendar
from tkinter import filedialog
from datetime import datetime as dt, timedelta


# ==================== LOGGING ====================
def setup_logger(name=__name__):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
        logger.addHandler(h)
    return logger

logger = setup_logger(__name__)


def setup_lot_logger(lot: str, dt_str: str) -> logging.Logger:
    log_dir = os.path.join('logs', lot)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{dt_str}_{lot}.log")
    lot_logger = logging.getLogger(f"lot.{lot}.{dt_str}")
    lot_logger.setLevel(logging.INFO)
    if not lot_logger.handlers:
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        lot_logger.addHandler(fh)
    return lot_logger


# ==================== ENUMS ====================
class PlaceState(Enum):
    EMPTY  = "EMPTY"   # รอวาง
    PLACED = "PLACED"  # วางแล้ว (เขียว)


# ==================== CONFIG ====================
@dataclass
class ROIBox:
    x: int
    y: int
    w: int
    h: int
    slots: int


class DetectionConfig:
    def __init__(self, config_path="config.json"):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)["DetectionConfig"]
        self.required_confirmations = cfg["required_confirmations"]
        self.delay_frames           = cfg["delay_frames"]
        self.min_object_area        = cfg["min_object_area"]
        self.max_object_area        = cfg["max_object_area"]
        self.expand_roi             = cfg["expand_roi"]
        self.threshold_value        = cfg["threshold_value"]


class SlotConfiguration:
    def __init__(self):
        with open('config.json', 'r', encoding='utf-8') as f:
            slot_cfg = json.load(f)['slots']
        # โหลดเฉพาะ 3 rows (ไม่มี NG row)
        self.boxes = [
            ROIBox(x=b.get('x', 0), y=b.get('y', 0),
                   w=b.get('w', 0), h=b.get('h', 0),
                   slots=b.get('slots', 0))
            for b in slot_cfg.get('boxes', [])[:3]
        ]
        # slot_start[i] = หมายเลข slot แรกของแถว i (อ่านจาก config)
        self.slot_start = slot_cfg.get('slot_start', [1, 21, 41])


class DatabaseConfig:
    MYSQL_CONFIG = {
        'host':      '172.18.106.100',
        'user':      'si',
        'password':  '123456',
        'database':  'koythee_db',
        'pool_name': 'pickpilot_pool',
        'pool_size': 5
    }


# ==================== HELPERS ====================
def play_alarm_beep():
    try:
        winsound.Beep(2300, 800)
    except Exception as e:
        logger.error(f"Beep: {e}")


# ==================== PLACE MODE DETECTION ====================
class PlaceModeDetectionSystem:
    """
    Place Mode — 3 แถว × 20 slots = 60 slots (Left-to-Right)
    ────────────────────────────────────────────────────────
    • ถาดว่างก่อน Start
    • วางทีละแผ่นตามลำดับ slot 1 → 2 → ... → 60
    • วางไม่ครบ 60 ก็ได้ — กด Stop เมื่อวางครบ แล้วกด Finished
    • วาง 2 แผ่นพร้อมกัน → ALARM + หยุดทันที → ต้อง Reset
    • วางผิดลำดับ         → ALARM + หยุดทันที → ต้อง Reset
    ────────────────────────────────────────────────────────
    """

    def __init__(self):
        self.config      = DetectionConfig()
        self.slot_config = SlotConfiguration()

        # สร้าง ROI ทุก slot ไว้ล่วงหน้า (60 slots จาก config)
        self.all_slots:   List[int]             = []
        self.slot_roi:    Dict[int, Tuple]      = {}
        self.states:      Dict[int, PlaceState] = {}
        self.current_index: int  = 1
        self.placed_count:  int  = 0     # นับจำนวนที่วางแล้ว
        self.initialized:   bool = False

        self.place_counter:   int  = 0
        self.frame_counter:   int  = 0
        self.alarm_triggered: bool = False

        self.alarm_callback:        Optional[Callable] = None
        self.ui_update_callback:    Optional[Callable] = None
        self.state_change_callback: Optional[Callable] = None
        self.lot_logger:            Optional[logging.Logger] = None

        self._build_full_slot_grid()

    @property
    def Box(self) -> List:
        return [[b.x, b.y, b.w, b.h] for b in self.slot_config.boxes[:3]]

    # ── callbacks ────────────────────────────────────────────────────────
    def set_alarm_callback(self, cb):        self.alarm_callback        = cb
    def set_ui_update_callback(self, cb):    self.ui_update_callback    = cb
    def set_state_change_callback(self, cb): self.state_change_callback = cb

    def _notify_state_change(self, slot_num, state):
        if self.state_change_callback:
            try: self.state_change_callback(slot_num, state)
            except Exception as e: logger.error(f"state_change cb: {e}")

    def _trigger_ui_update(self):
        if self.ui_update_callback:
            try: self.ui_update_callback()
            except Exception as e: logger.error(f"ui_update cb: {e}")

    # ── preprocessing ────────────────────────────────────────────────────
    def preprocess_frame(self, frame):
        if frame is None:
            return None
        try:
            # แปลงเป็น grayscale ก่อน threshold (รองรับทั้ง Mono8 และ BGR)
            gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(
                gray, self.config.threshold_value, 255, cv2.THRESH_BINARY_INV)
            k = np.ones((3, 3), np.uint8)
            return gray, cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
        except Exception as e:
            logger.error(f"preprocess: {e}")
            return None

    # ── build full slot grid (3 rows × 20 slots = 60) ────────────────
    def _build_full_slot_grid(self):
        """สร้าง ROI ครบ 60 slots — Left-to-Right ทุกแถว"""
        self.slot_roi.clear()
        self.all_slots.clear()

        for row_idx in range(3):
            roi_box    = self.slot_config.boxes[row_idx]
            n_slots    = roi_box.slots
            slot_start = self.slot_config.slot_start[row_idx]
            slot_w     = roi_box.w // n_slots

            for i in range(n_slots):
                slot_num = slot_start + i
                x = roi_box.x + i * slot_w   # Left-to-Right
                self.slot_roi[slot_num] = (x, roi_box.y, slot_w, roi_box.h)
                self.all_slots.append(slot_num)

        self.all_slots = sorted(self.all_slots)

    # ── reset state (ไม่ rebuild ROI) ───────────────────────────────────
    def _reset_state(self):
        self.states.clear()
        self.current_index   = self.all_slots[0] if self.all_slots else 1
        self.placed_count    = 0
        self.place_counter   = 0
        self.frame_counter   = 0
        self.alarm_triggered = False
        self.initialized     = False

    # ── initialize ───────────────────────────────────────────────────────
    def initialize_detection(self, initial_frame) -> bool:
        """ตรวจว่าถาดว่าง แล้ว init states ทุก slot เป็น EMPTY"""
        try:
            result = self.preprocess_frame(initial_frame)
            if result is None:
                return False
            _, thresh = result

            for row_idx in range(3):
                x, y, w, h = self.Box[row_idx]
                roi         = thresh[y:y+h, x:x+w]
                cnts, _     = cv2.findContours(
                    roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    area = cv2.contourArea(c)
                    if self.config.min_object_area < area < self.config.max_object_area:
                        logger.warning(f"Tray not empty (row {row_idx})")
                        return False

            self._reset_state()
            for slot_num in self.all_slots:
                self.states[slot_num] = PlaceState.EMPTY

            self.initialized = True
            self._trigger_ui_update()

            logger.info(f"Initialized: {len(self.all_slots)} slots ready, "
                        f"first={self.current_index}")
            if self.lot_logger:
                self.lot_logger.info("=" * 50)
                self.lot_logger.info("PLACE MODE STARTED")
                self.lot_logger.info(f"Total Slots Ready : {len(self.all_slots)}")
                self.lot_logger.info("=" * 50)
            return True

        except Exception as e:
            logger.error(f"initialize_detection: {e}")
            return False

    # ── contour helpers ──────────────────────────────────────────────────
    def _has_object_in_slot(self, thresh, slot_num: int) -> bool:
        if slot_num not in self.slot_roi:
            return False
        x, y, w, h = self.slot_roi[slot_num]
        exp        = self.config.expand_roi
        img_h, img_w = thresh.shape[:2]
        x0 = max(0, x - exp)
        x1 = min(img_w, x + w + exp)
        y0 = max(0, y)
        y1 = min(img_h, y + h)
        roi = thresh[y0:y1, x0:x1]
        try:
            cnts, _ = cv2.findContours(
                roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if self.config.min_object_area < cv2.contourArea(c) < self.config.max_object_area:
                    return True
        except Exception as e:
            logger.error(f"contour slot {slot_num}: {e}")
        return False

    def _get_newly_detected(self, thresh) -> List[int]:
        """คืน EMPTY slots ที่พบ object"""
        return [
            s for s in self.all_slots
            if self.states.get(s) == PlaceState.EMPTY
            and self._has_object_in_slot(thresh, s)
        ]

    # ── state machine ────────────────────────────────────────────────────
    def update_state_machine(self, thresh) -> str:
        """
        Returns: "NOT_INITIALIZED" | "RUNNING" | "ALARM"
        (ไม่มี COMPLETED — ผู้ใช้กด Stop เอง)

        ALARM logic:
          - scan MULTI_PLACE ทุก frame รวมถึงช่วง delay
          - detect >= 2  → ALARM MULTI_PLACE
          - detect == 1 ผิด slot → ALARM WRONG_ORDER
          - detect == 1 ถูก slot → confirm → PLACED
        """
        if not self.initialized:
            return "NOT_INITIALIZED"
        if self.alarm_triggered:
            return "ALARM"

        try:
            # scan MULTI_PLACE ก่อนเสมอ แม้อยู่ใน delay
            detected = self._get_newly_detected(thresh)

            if len(detected) >= 2:
                self._trigger_alarm("MULTI_PLACE", detected, self.current_index)
                return "ALARM"

            # frame delay หลัง PLACED
            if self.frame_counter > 0:
                self.frame_counter -= 1
                return "RUNNING"

            if not detected:
                self.place_counter = 0
                return "RUNNING"

            slot = detected[0]

            if slot != self.current_index:
                self._trigger_alarm("WRONG_ORDER", [slot], self.current_index)
                return "ALARM"

            # ถูกต้อง → confirm
            self.place_counter += 1
            if self.place_counter >= self.config.required_confirmations:
                placed = self.current_index
                self.states[placed] = PlaceState.PLACED
                self.placed_count  += 1

                logger.info(f"Slot {placed} PLACED  (total={self.placed_count})")
                if self.lot_logger:
                    self.lot_logger.info(
                        f"[PLACE] Slot {placed} PLACED  (total={self.placed_count})")
                self._notify_state_change(placed, PlaceState.PLACED)

                # ไปตำแหน่งถัดไป (slot ที่ยัง EMPTY)
                remaining = [s for s in self.all_slots
                             if self.states.get(s) == PlaceState.EMPTY]
                if remaining:
                    self.current_index = remaining[0]

                self.place_counter = 0
                self.frame_counter = self.config.delay_frames
                self._trigger_ui_update()

            return "RUNNING"

        except Exception as e:
            logger.error(f"state machine: {e}")
            return "RUNNING"

    def _trigger_alarm(self, alarm_type: str, slots: List[int], expected: int):
        self.alarm_triggered = True
        self.place_counter   = 0
        logger.warning(f"ALARM [{alarm_type}] detected={slots} expected={expected}")
        if self.lot_logger:
            self.lot_logger.warning(
                f"[ALARM] {alarm_type} | detected={slots} | expected={expected}")
        self._trigger_ui_update()
        if self.alarm_callback:
            try:
                self.alarm_callback(alarm_type=alarm_type,
                                    slots=slots, expected=expected)
            except Exception as e:
                logger.error(f"alarm cb: {e}")

    # ── summary ──────────────────────────────────────────────────────────
    def get_progress(self) -> Tuple[int, int]:
        """(placed, current_slot)"""
        return self.placed_count, self.current_index

    def get_detection_summary(self) -> Dict:
        return {
            "placed_count":  self.placed_count,
            "ok_count":      self.placed_count,
            "ng_count":      0,
            "broken_count":  0,
            "states":        {k: v.value for k, v in self.states.items()}
        }

    # ── draw ──────────────────────────────────────────────────────────────
    def draw_status(self, frame):
        if not self.initialized:
            return frame
        try:
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            for slot_num in self.all_slots:
                if slot_num not in self.slot_roi:
                    continue
                x, y, w, h = self.slot_roi[slot_num]
                state       = self.states.get(slot_num, PlaceState.EMPTY)

                if state == PlaceState.PLACED:
                    color, thick = (50, 205, 50), 3           # Green
                elif slot_num == self.current_index:
                    color = (0, 0, 255) if self.alarm_triggered else (0, 255, 255)
                    thick = 4                                  # Cyan / Red
                else:
                    color, thick = (80, 80, 80), 1            # Gray

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, thick)

                text        = str(slot_num)
                fs, ft      = 0.55, 2
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
                tx = x + (w - tw) // 2
                ty = y + h // 2 + th // 2
                cv2.rectangle(frame, (tx-2, ty-th-2), (tx+tw+2, ty+2), (0, 0, 0), -1)
                cv2.putText(frame, text, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, color, ft)

            cv2.putText(frame,
                        f"SLOT: {self.current_index}  PLACED: {self.placed_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
            if self.alarm_triggered:
                cv2.putText(frame, "!! ALARM - RESET REQUIRED !!",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
            return frame
        except Exception as e:
            logger.error(f"draw: {e}")
            return frame


# ==================== UI MANAGER ====================
class UIManager:
    def __init__(self, root, detector, time_tracker, data_manager):
        self.root         = root
        self.detector     = detector
        self.time_tracker = time_tracker
        self.data_manager = data_manager

        self.status_boxes:  Dict = {}
        self.input_entries: Dict = {}

        self.video_label:                Optional[tk.Label]        = None
        self.system_status_label:        Optional[tk.Label]        = None
        self.placed_label:               Optional[tk.Label]        = None   # PLACED count
        self.progress_bar:               Optional[ttk.Progressbar] = None
        self.start_label:             Optional[tk.Label]        = None
        self.stop_label:              Optional[tk.Label]        = None
        self.duration_label:             Optional[tk.Label]        = None
        self.current_index_status_label: Optional[tk.Label]        = None
        self.start_btn:                  Optional[tk.Button]       = None
        self.stop_btn:                   Optional[tk.Button]       = None
        self.finished_btn:               Optional[tk.Button]       = None
        self.reset_btn:                  Optional[tk.Button]       = None
        self.traveler_entry:             Optional[tk.Entry]        = None
        self.opt_id_entry:               Optional[tk.Entry]        = None

    def setup_ui(self):
        main = tk.Frame(self.root, bg='#F0F2F5')
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        main.grid_columnconfigure(0, weight=35, minsize=400)
        main.grid_columnconfigure(1, weight=30, minsize=400)
        main.grid_columnconfigure(2, weight=35, minsize=400)
        main.grid_rowconfigure(0, weight=1)
        self._setup_video_frame(main)
        self._setup_status_display(main)
        self._setup_control_panel(main)

    def _setup_video_frame(self, parent):
        f = tk.Frame(parent, bg='#FFFFFF', relief=tk.SOLID, bd=1)
        f.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        self.video_label = tk.Label(f, bg='#FFFFFF',
                                    text="Camera Feed Loading...",
                                    fg='#555555', font=('Arial', 16))
        self.video_label.pack(expand=True, fill=tk.BOTH)

    def _setup_status_display(self, parent):
        frame = tk.Frame(parent, bg='#FFFFFF', relief=tk.SOLID, bd=1)
        frame.grid(row=0, column=1, sticky='nsew', padx=5)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=0)
        frame.grid_columnconfigure(0, weight=1)

        upper = tk.Frame(frame, bg='#FFFFFF')
        upper.grid(row=0, column=0, sticky="nsew")
        tk.Label(upper, text="SLOT STATUS",
                 font=('Arial', 14, 'bold'), fg='#1565C0', bg='#FFFFFF').pack(pady=10)

        canvas = tk.Canvas(upper, bg='#FFFFFF', highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        cf = tk.Frame(canvas, bg='#FFFFFF')
        cf.bind("<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=cf, anchor="nw")
        self._create_status_boxes(cf)

        bot = tk.Frame(frame, bg='#E3F2FD', relief=tk.FLAT, bd=0)
        bot.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        self.current_index_status_label = tk.Label(
            bot, text="PLACED : 0  |  NEXT SLOT : -",
            font=('Arial', 14, 'bold'), fg="#1565C0", bg='#E3F2FD')
        self.current_index_status_label.pack(pady=5)

    def _create_status_boxes(self, parent):
        BOX_W, BOX_H, SP = 16, 220, 3
        SLOTS_PER_ROW    = 20
        SLOT_START       = [1, 21, 41]
        center = tk.Frame(parent, bg='#FFFFFF')
        center.pack(expand=True)
        for row_idx in range(2, -1, -1):
            row_frame = tk.Frame(center, bg='#FFFFFF')
            row_frame.pack(pady=SP)
            start = SLOT_START[row_idx]
            for col in range(SLOTS_PER_ROW):
                slot_num = start + col
                bf = tk.Frame(row_frame, bg='#CFD8DC',
                              width=BOX_W, height=BOX_H,
                              relief=tk.RAISED, bd=1)
                bf.pack(side=tk.LEFT, padx=SP)
                bf.pack_propagate(False)
                lbl = tk.Label(bf, text=str(slot_num),
                               font=('Arial', 8, 'bold'),
                               fg='#37474F', bg='#CFD8DC')
                lbl.pack(expand=True)
                self.status_boxes[slot_num] = {
                    'frame': bf, 'label': lbl, 'current_state': None}

    def _setup_control_panel(self, parent):
        ctrl = tk.Frame(parent, bg='#FFFFFF', relief=tk.SOLID, bd=1)
        ctrl.grid(row=0, column=2, sticky='nsew', padx=(5, 0))
        ctrl.grid_columnconfigure(0, weight=1)
        self._setup_form(ctrl)
        self._setup_time_section(ctrl)
        self._setup_buttons(ctrl)
        self._setup_bottom_status(ctrl)

    def _setup_form(self, parent):
        form = tk.Frame(parent, bg='#F5F5F5', relief=tk.SOLID, bd=1)
        form.pack(pady=10, padx=10, fill=tk.X)

        LF = ('Arial', 10, 'bold')
        EF = ('Arial', 10)

        def _row(label_text, key, bind_return=None):
            f = tk.Frame(form, bg='#F5F5F5')
            f.pack(fill=tk.X, padx=10, pady=8)
            tk.Label(f, text=label_text, font=LF, fg='#333333', bg='#F5F5F5',
                     width=13, anchor='w').pack(side=tk.LEFT)
            e = tk.Entry(f, font=EF, bg='#FFFFFF', fg='#111111',
                         insertbackground='#1565C0', relief=tk.SOLID, bd=1)
            e.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6)
            if bind_return:
                e.bind("<Return>", bind_return)
            self.input_entries[key] = e
            return e

        self.traveler_entry = _row("Lot Number:", 'traveler_lot',
                                   bind_return=self._on_lot_enter)
        self.traveler_entry.focus()
        self.opt_id_entry = _row("OPT ID:", 'opt_id',
                                 bind_return=self._on_opt_id_enter)

    def _on_lot_enter(self, event):
        if event.widget.get().strip():
            self.opt_id_entry.focus()

    def _on_opt_id_enter(self, event):
        if event.widget.get().strip():
            if self.start_btn and str(self.start_btn['state']) != 'disabled':
                self.start_btn.focus()

    def _setup_time_section(self, parent):
        f = tk.Frame(parent, bg='#F5F5F5', relief=tk.SOLID, bd=1)
        f.pack(pady=10, padx=10, fill=tk.X)
        tk.Label(f, text="⏱️ TIME WORKING",
                 font=('Arial', 11, 'bold'), fg='#1565C0', bg='#F5F5F5').pack(pady=5)
        for text, attr, color in [
            ("Start:",    'start_label',    '#2E7D32'),
            ("Stop:",     'stop_label',     '#C62828'),
            ("Duration:", 'duration_label', '#1565C0'),
        ]:
            row = tk.Frame(f, bg='#F5F5F5')
            row.pack(fill=tk.X, padx=10, pady=3)
            tk.Label(row, text=text, font=('Arial', 11, 'bold'),
                     fg='#333333', bg='#F5F5F5',
                     width=12, anchor='w').pack(side=tk.LEFT)
            lbl = tk.Label(row,
                           text="--:--:--" if "Duration" not in text else "00:00:00",
                           font=('Arial', 11, 'bold'), fg=color,
                           bg='#FFFFFF', relief=tk.SOLID, bd=1, anchor='w', padx=5)
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)
            setattr(self, attr, lbl)

    def _setup_buttons(self, parent):
        bf = tk.Frame(parent, bg='#FFFFFF')
        bf.pack(pady=10, padx=10, fill=tk.X)

        self.start_btn = tk.Button(
            bf, text="▶️  Start",
            bg='#1565C0', fg='white', font=('Arial', 13, 'bold'),
            height=2, relief=tk.FLAT, bd=0, cursor='hand2')
        self.start_btn.pack(fill=tk.X, pady=2)

        self.stop_btn = tk.Button(
            bf, text="⏹  Stop",
            bg='#E53935', fg='white', font=('Arial', 13, 'bold'),
            height=2, relief=tk.FLAT, bd=0, cursor='hand2', state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=2)

        self.finished_btn = tk.Button(
            bf, text="✅  Finished",
            bg='#2E7D32', fg='white', font=('Arial', 13, 'bold'),
            height=2, relief=tk.FLAT, bd=0, cursor='hand2', state=tk.DISABLED)
        self.finished_btn.pack(fill=tk.X, pady=2)

        self.reset_btn = tk.Button(
            bf, text="🔄  Reset",
            bg='#E65100', fg='white', font=('Arial', 13, 'bold'),
            height=2, relief=tk.FLAT, bd=0, cursor='hand2')
        self.reset_btn.pack(fill=tk.X, pady=2)

    def _setup_bottom_status(self, parent):
        bf = tk.Frame(parent, bg='#FFFFFF')
        bf.pack(pady=10, padx=10, fill=tk.X, side='bottom')

        self.placed_label = tk.Label(
            bf, text="Placed: 0",
            font=('Arial', 14, 'bold'), fg='#2E7D32', bg='#FFFFFF')
        self.placed_label.pack(pady=4)

        self.system_status_label = tk.Label(
            bf, text="Status: Ready",
            font=('Arial', 11, 'bold'), fg='#1565C0', bg='#FFFFFF')
        self.system_status_label.pack()

        leg = tk.Frame(bf, bg='#FFFFFF')
        leg.pack(fill=tk.X, pady=5)
        for text, color in [("■ WAITING", '#0277BD'),
                             ("■ PLACED",  '#2E7D32'),
                             ("■ ALARM",   '#C62828')]:
            tk.Label(leg, text=text, font=('Arial', 9, 'bold'),
                     fg=color, bg='#FFFFFF').pack(side=tk.LEFT, padx=5)

    # ── update methods ────────────────────────────────────────────────────
    def update_status_boxes(self, changed_slots=None):
        if not self.detector.initialized:
            for box in self.status_boxes.values():
                box['frame'].config(bg='#CFD8DC', relief='raised', bd=1)
                box['label'].config(bg='#CFD8DC', fg='#37474F')
                box['current_state'] = None
            return

        slots = changed_slots if changed_slots else list(self.detector.all_slots)

        for slot_num in slots:
            if slot_num not in self.status_boxes:
                continue
            state = self.detector.states.get(slot_num, PlaceState.EMPTY)
            box   = self.status_boxes[slot_num]

            if box.get('current_state') == state and changed_slots is not None:
                continue

            if state == PlaceState.PLACED:
                bg, fg, relief = '#43A047', 'white', 'sunken'   # เขียวเข้ม
            elif slot_num == self.detector.current_index:
                if self.detector.alarm_triggered:
                    bg, fg, relief = '#E53935', 'white', 'raised'  # แดง
                else:
                    bg, fg, relief = '#0277BD', 'white', 'raised'  # น้ำเงิน (NEXT)
            else:
                bg, fg, relief = '#CFD8DC', '#37474F', 'raised'    # เทาอ่อน

            box['frame'].config(bg=bg, relief=relief, bd=1)
            box['label'].config(bg=bg, fg=fg)
            box['current_state'] = state

    def update_counter(self):
        if self.detector.initialized:
            placed, next_slot = self.detector.get_progress()
            self.placed_label.config(text=f"Placed: {placed}")
            self.current_index_status_label.config(
                text=f"PLACED : {placed}  |  NEXT SLOT : {next_slot}")
        else:
            self.placed_label.config(text="Placed: 0")
            self.current_index_status_label.config(
                text="PLACED : 0  |  NEXT SLOT : -")

    def update_time_display(self):
        if self.time_tracker.start_time:
            self.duration_label.config(
                text=self.time_tracker.get_duration_formatted())

    def update_all_components(self):
        try:
            self.update_status_boxes()
            self.update_counter()
        except Exception as e:
            logger.error(f"ui update: {e}")

    def get_form_data(self) -> Dict[str, str]:
        return {k: e.get().strip() for k, e in self.input_entries.items()}

    def validate_form_data(self) -> bool:
        data = self.get_form_data()
        missing = []
        if not data.get('traveler_lot'): missing.append('Lot Number')
        if not data.get('opt_id'):       missing.append('OPT ID')
        if missing:
            play_alarm_beep()
            messagebox.showwarning(
                "⚠️ ข้อมูลไม่ครบ",
                "กรุณากรอก:\n\n" + "\n".join(f"• {m}" for m in missing),
                parent=self.root)
            return False
        return True

    def clear_form(self):
        for entry in self.input_entries.values():
            entry.delete(0, tk.END)
        self.traveler_entry.focus()

    def reset_time_display(self):
        self.start_label.config(text="--:--:--")
        self.stop_label.config(text="--:--:--")
        self.duration_label.config(text="00:00:00")
        self.current_index_status_label.config(
            text="PLACED : 0  |  NEXT SLOT : -")


# ==================== DATA MANAGER ====================
class DataManager:
    def __init__(self):
        self.log_folder = "inspection_logs"
        self.mysql_pool = None
        os.makedirs(self.log_folder, exist_ok=True)
        self._init_mysql()

    def _init_mysql(self):
        try:
            self.mysql_pool = pooling.MySQLConnectionPool(**DatabaseConfig.MYSQL_CONFIG)
            logger.info("MySQL pool ready")
        except Error as e:
            logger.error(f"MySQL pool: {e}")
            self.mysql_pool = None

    def save_mysql(self, data: Dict) -> bool:
        if not self.mysql_pool:
            return False
        conn = cur = None
        try:
            fd = data.get('form_data', {})
            dd = data.get('detection_data', {})
            td = data.get('time_data', {})
            row = (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                fd.get('traveler_lot'),     # travlot
                None,                       # polishlot (ไม่ใช้)
                None,                       # productsize (ไม่ใช้)
                fd.get('opt_id'),           # nrinspector
                str(dd.get('placed_count', 0)),  # glassinput = จำนวนที่วางจริง
                dd.get('ok_count', 0),      # ok
                0,                          # ng
                0,                          # other
                td.get('start_time'),
                td.get('stop_time'),
            )
            conn = self.mysql_pool.get_connection()
            cur  = conn.cursor()
            cur.execute(
                "INSERT INTO fp_clvi_pskipnrins "
                "(datetime,travlot,polishlot,productsize,nrinspector,glassinput,"
                " ok,ng,other,starttime,endtime) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", row)
            conn.commit()
            logger.info("MySQL saved")
            return True
        except Error as e:
            logger.error(f"MySQL save: {e}")
            if conn: conn.rollback()
            return False
        finally:
            if cur:  cur.close()
            if conn: conn.close()

    def save_inspection_data(self, form_data, time_data, detection_data) -> bool:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        data = {"timestamp": ts, "form_data": form_data,
                "time_data": time_data, "detection_data": detection_data}
        # ok1  = self.save_mysql(data)
        ok1 = True
        ok2  = False
        try:
            with open(f"{self.log_folder}/inspection_{ts}.json",
                      'w', encoding='utf-8') as f:
                import json as _json
                _json.dump(data, f, indent=4, ensure_ascii=False)
            ok2 = True
        except Exception as e:
            logger.error(f"json save: {e}")
        return ok1 or ok2


# ==================== TIME TRACKER ====================
class TimeTracker:
    def __init__(self):
        self.start_time: Optional[datetime] = None
        self.stop_time:  Optional[datetime] = None

    def start(self) -> str:
        self.start_time = datetime.now()
        self.stop_time  = None
        return self.start_time.strftime("%H:%M:%S")

    def stop(self) -> str:
        self.stop_time = datetime.now()
        return self.stop_time.strftime("%H:%M:%S")

    def get_duration(self) -> float:
        if not self.start_time:
            return 0.0
        end = self.stop_time or datetime.now()
        return max(0.0, (end - self.start_time).total_seconds())

    def get_duration_formatted(self) -> str:
        s = int(self.get_duration())
        return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

    def reset(self):
        self.start_time = self.stop_time = None


# ==================== MAIN APPLICATION ====================
class DetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Place Monitoring System Ver.1.0")
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg='#F0F2F5')

        self._create_top_bar()

        self.camera           = Camera()
        self.detector         = PlaceModeDetectionSystem()
        self.time_tracker     = TimeTracker()
        self.data_manager     = DataManager()
        self.ui_manager       = UIManager(
            self.root, self.detector, self.time_tracker, self.data_manager)

        self.is_running:          bool = False
        self.camera_running:      bool = True
        self.current_frame               = None
        self.lot_logger:          Optional[logging.Logger] = None
        self.update_timer_id               = None
        self._alarm_shown:        bool = False

        self._setup_callbacks()
        self.ui_manager.setup_ui()
        self._bind_buttons()
        self._init_camera()
        logger.info("DetectionApp ready")

    def _create_top_bar(self):
        bar = tk.Frame(self.root, bg='#1565C0', height=60,
                       relief=tk.FLAT, bd=0)
        bar.pack(side=tk.TOP, fill=tk.X)
        bar.pack_propagate(False)
        tk.Label(bar, text="Smart Place Monitoring System Ver.1.0",
                 font=('Arial', 16, 'bold'), fg='white', bg='#1565C0'
                 ).pack(side=tk.LEFT, padx=15, pady=8)
        for text, bg, cmd in [
            ("✕",  '#E53935', self.on_closing),
            ("―",  '#FB8C00', self.minimize_window),
            ("📄", '#1E88E5', self.show_report),
        ]:
            tk.Button(bar, text=text, font=('Arial', 16, 'bold'), fg='white',
                      bg=bg, bd=0, width=3, cursor='hand2',
                      command=cmd).pack(side=tk.RIGHT, padx=5, pady=5)

    def _setup_callbacks(self):
        self.detector.set_alarm_callback(self._on_alarm)
        self.detector.set_ui_update_callback(self.schedule_ui_update)
        self.detector.set_state_change_callback(self._on_state_change)

    def _bind_buttons(self):
        self.ui_manager.start_btn.config(command=self.start_detection)
        self.ui_manager.stop_btn.config(command=self.stop_detection)
        self.ui_manager.finished_btn.config(command=self.finish_and_save)
        self.ui_manager.reset_btn.config(command=self.reset_system)

    def schedule_ui_update(self):
        try:
            self.root.after(0, self.ui_manager.update_all_components)
        except Exception as e:
            logger.error(f"schedule_ui_update: {e}")

    def _on_state_change(self, slot_num, new_state):
        try:
            self.root.after(0, lambda: self.ui_manager.update_status_boxes([slot_num]))
        except Exception as e:
            logger.error(f"state_change handler: {e}")

    def _on_alarm(self, alarm_type: str, slots: List[int], expected: int):
        self.root.after(0, lambda: self._show_alarm(alarm_type, slots, expected))

    def _show_alarm(self, alarm_type: str, slots: List[int], expected: int):
        if self._alarm_shown:
            return
        self._alarm_shown = True

        self._stop_internal()   # หยุด detection ทันที

        play_alarm_beep()

        if alarm_type == "MULTI_PLACE":
            title = "⚠️ วางหลายชิ้นพร้อมกัน!"
            msg   = (f"❌ ตรวจพบชิ้นงาน {len(slots)} ตำแหน่งพร้อมกัน!\n\n"
                     f"🔴 ตำแหน่งที่พบ : {slots}\n"
                     f"✅ ควรวางที่    : Slot {expected}\n\n"
                     f"กรุณากด 🔄 Reset และเริ่มใหม่")
        else:
            title = "⚠️ วางผิดลำดับ!"
            msg   = (f"❌ วางที่ช่อง {slots[0]} "
                     f"แต่ต้องวางช่อง {expected} ก่อน!\n\n"
                     f"กรุณากด 🔄 Reset และเริ่มใหม่")

        messagebox.showwarning(title, msg, parent=self.root)
        self.ui_manager.system_status_label.config(
            text="⚠️ ALARM - กรุณา Reset", fg='#C62828')

        # disable Start, Stop, Finished
        self.ui_manager.start_btn.config(state=tk.DISABLED)
        self.ui_manager.stop_btn.config(state=tk.DISABLED)
        self.ui_manager.finished_btn.config(state=tk.DISABLED)
        self.ui_manager.update_all_components()

    # ── camera ────────────────────────────────────────────────────────────
    def _init_camera(self):
        try:
            if self.camera.open(1):
                self.ui_manager.system_status_label.config(
                    text="Status: Ready – กรอกข้อมูลแล้วกด Start", fg='#1565C0')
                threading.Thread(target=self._video_loop, daemon=True).start()
            else:
                self.ui_manager.system_status_label.config(
                    text="Status: Camera Error", fg='#C62828')
                play_alarm_beep()
                messagebox.showerror("❌ กล้องไม่ตอบสนอง",
                                     "ไม่สามารถเชื่อมต่อกล้องได้",
                                     parent=self.root)
        except Exception as e:
            logger.error(f"init_camera: {e}")

    def _video_loop(self):
        while self.camera_running:
            try:
                frame = self.camera.get_frame()
                if frame is None:
                    continue
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                self.current_frame = frame

                if self.is_running and self.detector.initialized:
                    result = self.detector.preprocess_frame(frame)
                    if result:
                        _, thresh = result
                        self.detector.update_state_machine(thresh)
                        # ไม่มี COMPLETED — ผู้ใช้กด Stop เอง

                display = self.detector.draw_status(frame.copy())
                if not self.detector.initialized:
                    if len(display.shape) == 2:
                        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
                    for x, y, w, h in self.detector.Box:
                        cv2.rectangle(display, (x, y), (x+w, y+h), (60, 60, 60), 2)

                self._display_frame(display)
            except Exception as e:
                logger.error(f"video loop: {e}")
                break

    def _display_frame(self, frame):
        try:
            resized = cv2.resize(frame, (550, 800))
            rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            photo   = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.root.after(0, self._update_video_label, photo)
        except Exception as e:
            logger.error(f"display frame: {e}")

    def _update_video_label(self, photo):
        self.ui_manager.video_label.configure(image=photo, text="")
        self.ui_manager.video_label.image = photo

    # ── START ─────────────────────────────────────────────────────────────
    def start_detection(self):
        if not self.ui_manager.validate_form_data():
            return

        if self.current_frame is None:
            play_alarm_beep()
            messagebox.showwarning("⚠️ ไม่พบสัญญาณกล้อง",
                                   "กรุณาตรวจสอบการเชื่อมต่อกล้อง",
                                   parent=self.root)
            return

        if not self.detector.initialize_detection(self.current_frame):
            play_alarm_beep()
            messagebox.showerror(
                "❌ ถาดไม่ว่าง",
                "ตรวจพบชิ้นงานในถาด\n\nกรุณาเอาออกให้หมดก่อนกด Start",
                parent=self.root)
            return

        form_data    = self.ui_manager.get_form_data()
        traveler_lot = form_data.get('traveler_lot', 'UNKNOWN')
        dt_str       = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.lot_logger          = setup_lot_logger(traveler_lot, dt_str)
        self.detector.lot_logger = self.lot_logger

        self.lot_logger.info("=" * 50)
        self.lot_logger.info("PLACE MODE STARTED")
        self.lot_logger.info(f"Lot Number : {traveler_lot}")
        self.lot_logger.info(f"OPT ID     : {form_data.get('opt_id', '-')}")
        self.lot_logger.info("=" * 50)

        self._alarm_shown = False
        self.is_running   = True

        self.ui_manager.start_btn.config(state=tk.DISABLED, bg='#90A4AE')
        self.ui_manager.stop_btn.config(state=tk.NORMAL)
        self.ui_manager.finished_btn.config(state=tk.DISABLED)
        self.ui_manager.system_status_label.config(
            text="Status: Running ▶️", fg='#1565C0')

        start_str = self.time_tracker.start()
        self.ui_manager.start_label.config(text=start_str)
        self._tick_timer()
        logger.info("Detection started")

    # ── STOP ──────────────────────────────────────────────────────────────
    def stop_detection(self):
        """พนักงานกด Stop — หยุด detection แต่ยังไม่บันทึก"""
        self._stop_internal()

        self.ui_manager.start_btn.config(state=tk.DISABLED, bg='#90A4AE')
        self.ui_manager.stop_btn.config(state=tk.DISABLED)
        self.ui_manager.finished_btn.config(state=tk.NORMAL)
        self.ui_manager.system_status_label.config(
            text="Status: Stopped – กด ✅ Finished เพื่อบันทึก", fg='#E65100')

        placed = self.detector.placed_count
        logger.info(f"Detection stopped by user. Placed={placed}")
        if self.lot_logger:
            self.lot_logger.info(f"[STOP] User stopped. Placed={placed}")

    def _stop_internal(self):
        if not self.is_running:
            return
        self.is_running = False
        stop_str = self.time_tracker.stop()
        self.ui_manager.stop_label.config(text=stop_str)
        if self.update_timer_id:
            self.root.after_cancel(self.update_timer_id)
            self.update_timer_id = None
        logger.info("Detection stopped")

    def _tick_timer(self):
        if self.is_running:
            self.ui_manager.update_time_display()
            self.update_timer_id = self.root.after(1000, self._tick_timer)

    # ── FINISHED → SAVE ───────────────────────────────────────────────────
    def finish_and_save(self):
        """บันทึกผลและ Reset"""
        placed = self.detector.placed_count
        duration = self.time_tracker.get_duration_formatted()

        if self.lot_logger:
            self.lot_logger.info("=" * 50)
            self.lot_logger.info("FINISHED")
            self.lot_logger.info(f"Placed   : {placed}")
            self.lot_logger.info(f"Duration : {duration}")
            self.lot_logger.info("=" * 50)

        saved = self._save_report(silent=True)

        msg  = "✅ บันทึกผลสำเร็จ!\n\n"
        msg += f"🔢 วางแล้ว : {placed} ชิ้น\n"
        msg += f"⏱️ เวลา   : {duration}\n"
        msg += "\n💾 บันทึก DB เรียบร้อย ✅" if saved else "\n⚠️ บันทึก DB ล้มเหลว"

        messagebox.showinfo("✅ Finished", msg, parent=self.root)
        self._reset_internal(silent=True)

    def _save_report(self, silent=False) -> bool:
        try:
            form_data = self.ui_manager.get_form_data()
            time_data = {
                "start_time": (self.time_tracker.start_time
                               .strftime("%Y-%m-%d %H:%M:%S")
                               if self.time_tracker.start_time else None),
                "stop_time":  (self.time_tracker.stop_time
                               .strftime("%Y-%m-%d %H:%M:%S")
                               if self.time_tracker.stop_time else None),
                "duration":   self.time_tracker.get_duration_formatted()
            }
            ok = self.data_manager.save_inspection_data(
                form_data, time_data, self.detector.get_detection_summary())
            if not ok and not silent:
                play_alarm_beep()
                messagebox.showerror("❌ บันทึกไม่สำเร็จ",
                                     "ไม่สามารถบันทึกได้",
                                     parent=self.root)
            return ok
        except Exception as e:
            logger.error(f"save_report: {e}")
            return False

    # ── RESET ─────────────────────────────────────────────────────────────
    def reset_system(self):
        if not messagebox.askyesno("⚠️ ยืนยัน Reset",
                                   "รีเซ็ตระบบ?\nข้อมูลที่ยังไม่บันทึกจะหายไป",
                                   parent=self.root):
            return
        self._reset_internal()

    def _reset_internal(self, silent=False):
        try:
            self._stop_internal()
            self.detector._reset_state()
            self.detector.initialized = False
            self._alarm_shown         = False
            self.lot_logger           = None
            self.detector.lot_logger  = None

            self.time_tracker.reset()
            self.ui_manager.clear_form()
            self.ui_manager.reset_time_display()
            self.ui_manager.update_all_components()

            self.ui_manager.start_btn.config(
                text="▶️  Start", bg='#1565C0', state=tk.NORMAL)
            self.ui_manager.stop_btn.config(state=tk.DISABLED)
            self.ui_manager.finished_btn.config(state=tk.DISABLED)
            self.ui_manager.system_status_label.config(
                text="Status: Ready – กรอกข้อมูลแล้วกด Start", fg='#1565C0')

            logger.info("System reset")
            if not silent:
                messagebox.showinfo("✅ Reset สำเร็จ", "พร้อมเริ่มงานใหม่",
                                    parent=self.root)
        except Exception as e:
            logger.error(f"reset: {e}")
            play_alarm_beep()
            messagebox.showerror("❌ Reset ล้มเหลว", str(e), parent=self.root)

    # ── WINDOW ────────────────────────────────────────────────────────────
    def minimize_window(self):
        self.root.iconify()

    def on_closing(self):
        if messagebox.askyesno("⚠️ ออกจากโปรแกรม",
                               "ต้องการปิดโปรแกรมหรือไม่?",
                               parent=self.root):
            try:
                self.is_running     = False
                self.camera_running = False
                if self.update_timer_id:
                    self.root.after_cancel(self.update_timer_id)
                self.camera.close()
                import time; time.sleep(0.4)
                self.root.destroy()
            except Exception as e:
                logger.error(f"closing: {e}")
                self.root.destroy()

    # ── REPORT WINDOW ─────────────────────────────────────────────────────
    def show_report(self):
        if not self.data_manager.mysql_pool:
            play_alarm_beep()
            messagebox.showerror("❌ DB Error",
                                 "ไม่สามารถเชื่อมต่อ Database ได้",
                                 parent=self.root)
            return

        win = tk.Toplevel(self.root)
        win.overrideredirect(True)
        win.configure(bg="#1565C0")
        W, H, B = 1400, 760, 2
        sw, sh  = win.winfo_screenwidth(), win.winfo_screenheight()
        win.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")
        win.grab_set()

        _d = {"x": 0, "y": 0}
        def _ds(e): _d["x"], _d["y"] = e.x_root, e.y_root
        def _dm(e):
            win.geometry(f"+{win.winfo_x()+e.x_root-_d['x']}"
                         f"+{win.winfo_y()+e.y_root-_d['y']}")
            _d["x"], _d["y"] = e.x_root, e.y_root

        outer = tk.Frame(win, bg="#1565C0")
        outer.pack(fill=tk.BOTH, expand=True, padx=B, pady=B)
        inner = tk.Frame(outer, bg="#F5F5F5")
        inner.pack(fill=tk.BOTH, expand=True)

        hdr = tk.Frame(inner, bg="#1565C0", height=48)
        hdr.pack(fill=tk.X); hdr.pack_propagate(False)
        hdr.bind("<ButtonPress-1>", _ds); hdr.bind("<B1-Motion>", _dm)
        tl = tk.Label(hdr, text="📄 Inspection Report",
                      font=("Arial", 14, "bold"), fg="white", bg="#1565C0")
        tl.pack(side=tk.LEFT, padx=15, pady=10)
        tl.bind("<ButtonPress-1>", _ds); tl.bind("<B1-Motion>", _dm)
        tk.Button(hdr, text="✕", font=("Arial", 13, "bold"), fg="white",
                  bg="#E53935", bd=0, width=4, cursor="hand2",
                  command=win.destroy).pack(side=tk.RIGHT, padx=6, pady=6)

        # ── calendar popup ─────────────────────────────────────────────
        _cs = {"popup": None}

        def _open_cal(anchor, dvar):
            if _cs["popup"]:
                try: _cs["popup"].destroy()
                except: pass
                _cs["popup"] = None
            try: cur = dt.strptime(dvar.get(), "%Y-%m-%d")
            except: cur = dt.now()
            nav = {"y": cur.year, "m": cur.month}
            pop = tk.Toplevel(win)
            pop.overrideredirect(True); pop.configure(bg="#FFFFFF")
            pop.attributes("-topmost", True); _cs["popup"] = pop
            win.update_idletasks()
            pop.geometry(f"+{anchor.winfo_rootx()}+"
                         f"{anchor.winfo_rooty()+anchor.winfo_height()+4}")
            nb = tk.Frame(pop, bg="#1565C0"); nb.pack(fill=tk.X)
            nl = tk.Label(nb, text="", font=("Arial", 10, "bold"),
                          fg="white", bg="#1565C0", width=16)
            nl.pack(side=tk.LEFT, expand=True, padx=4, pady=6)
            bk = dict(font=("Arial", 11, "bold"), fg="white", bg="#1565C0",
                      activebackground="#1976D2", bd=0, cursor="hand2", width=3)
            gf = tk.Frame(pop, bg="#FFFFFF", padx=6, pady=4); gf.pack()
            for ci, dn in enumerate(["Mo","Tu","We","Th","Fr","Sa","Su"]):
                fc = "#E53935" if dn=="Su" else "#1565C0" if dn=="Sa" else "#555555"
                tk.Label(gf, text=dn, font=("Arial", 8, "bold"),
                         fg=fc, bg="#FFFFFF", width=4).grid(row=0, column=ci, pady=(0, 2))
            dbs = []

            def rnd():
                y, m = nav["y"], nav["m"]
                nl.config(text=f"{calendar.month_abbr[m]}  {y}")
                for b in dbs: b.destroy()
                dbs.clear()
                ts, sel = dt.now().strftime("%Y-%m-%d"), dvar.get()
                fw, di  = calendar.monthrange(y, m)
                for idx in range(42):
                    dn2 = idx - fw + 1; rr = idx//7+1; cr = idx%7
                    if dn2 < 1 or dn2 > di:
                        lb = tk.Label(gf, text="", bg="#FFFFFF", width=4)
                        lb.grid(row=rr, column=cr); dbs.append(lb); continue
                    ds2 = f"{y:04d}-{m:02d}-{dn2:02d}"
                    if ds2 == sel:  bc, fc = "#1565C0", "white"
                    elif ds2 == ts: bc, fc = "#2E7D32", "white"
                    elif cr == 6:   bc, fc = "#FFFFFF", "#E53935"
                    elif cr == 5:   bc, fc = "#FFFFFF", "#1565C0"
                    else:           bc, fc = "#FFFFFF", "#222222"
                    b = tk.Button(gf, text=str(dn2), font=("Arial", 9), width=4,
                                  bg=bc, fg=fc, activebackground="#BBDEFB",
                                  activeforeground="#111111", bd=0, cursor="hand2",
                                  relief=tk.FLAT, command=lambda d=ds2: _pk(d))
                    b.grid(row=rr, column=cr, padx=1, pady=1); dbs.append(b)

            def _pv():
                nav["m"] = 12 if nav["m"] == 1 else nav["m"] - 1
                if nav["m"] == 12 and nav["y"] > 2000: nav["y"] -= 1
                rnd()

            def _nx():
                nav["m"] = 1 if nav["m"] == 12 else nav["m"] + 1
                if nav["m"] == 1: nav["y"] += 1
                rnd()

            def _pk(d):
                dvar.set(d); _cs["popup"] = None; pop.destroy()

            tk.Button(nb, text="◀", command=_pv, **bk).pack(side=tk.LEFT,  pady=4)
            tk.Button(nb, text="▶", command=_nx, **bk).pack(side=tk.RIGHT, pady=4)
            rnd(); pop.focus_set()

        # ── toolbar ────────────────────────────────────────────────────
        tb  = tk.Frame(inner, bg="#E3F2FD", pady=8); tb.pack(fill=tk.X, padx=10)
        lk  = dict(font=("Arial", 10, "bold"), fg="#1565C0", bg="#E3F2FD")
        ek  = dict(font=("Arial", 10), bg="#FFFFFF", fg="#111111",
                   insertbackground="#1565C0", relief=tk.SOLID, bd=1, state="readonly")
        tod = dt.now(); yes = tod - timedelta(days=1)

        tk.Label(tb, text="📅 From:", **lk).pack(side=tk.LEFT, padx=(0, 3))
        fv = tk.StringVar(value=yes.strftime("%Y-%m-%d"))
        fe = tk.Entry(tb, textvariable=fv, width=12, **ek)
        fe.pack(side=tk.LEFT, ipady=5, padx=(0, 2))
        tk.Button(tb, text="🗓", font=("Arial", 11), fg="#1565C0", bg="#BBDEFB",
                  activebackground="#90CAF9", bd=0, cursor="hand2",
                  command=lambda: _open_cal(fe, fv)
                  ).pack(side=tk.LEFT, ipady=3, padx=(0, 14))

        tk.Label(tb, text="To:", **lk).pack(side=tk.LEFT, padx=(0, 3))
        tv = tk.StringVar(value=tod.strftime("%Y-%m-%d"))
        te = tk.Entry(tb, textvariable=tv, width=12, **ek)
        te.pack(side=tk.LEFT, ipady=5, padx=(0, 2))
        tk.Button(tb, text="🗓", font=("Arial", 11), fg="#1565C0", bg="#BBDEFB",
                  activebackground="#90CAF9", bd=0, cursor="hand2",
                  command=lambda: _open_cal(te, tv)
                  ).pack(side=tk.LEFT, ipady=3, padx=(0, 14))

        rb = tk.Button(tb, text="🔄 โหลด", font=("Arial", 10, "bold"), fg="white",
                       bg="#1565C0", activebackground="#0D47A1",
                       bd=0, padx=12, cursor="hand2")
        rb.pack(side=tk.LEFT, ipady=5, padx=(0, 20))
        tk.Frame(tb, bg="#BDBDBD", width=2, height=28).pack(side=tk.LEFT, padx=4)
        tk.Label(tb, text="🔍 ค้นหา:", **lk).pack(side=tk.LEFT, padx=(8, 3))
        sv = tk.StringVar()
        se = tk.Entry(tb, textvariable=sv, width=24,
                      font=("Arial", 10), bg="#FFFFFF", fg="#111111",
                      insertbackground="#1565C0", relief=tk.SOLID, bd=1)
        se.pack(side=tk.LEFT, ipady=5, padx=(0, 14))
        eb = tk.Button(tb, text="⬇️ Export CSV", font=("Arial", 10, "bold"), fg="white",
                       bg="#2E7D32", activebackground="#1B5E20",
                       bd=0, padx=12, cursor="hand2")
        eb.pack(side=tk.LEFT, ipady=5)

        # ── treeview ───────────────────────────────────────────────────
        tf = tk.Frame(inner, bg="#F5F5F5")
        tf.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 0))
        sty = ttk.Style(); sty.theme_use("clam")
        sty.configure("R.Treeview", background="#FFFFFF", foreground="#111111",
                      rowheight=26, fieldbackground="#FFFFFF",
                      borderwidth=0, font=("Arial", 9))
        sty.configure("R.Treeview.Heading", background="#1565C0",
                      foreground="white", font=("Arial", 9, "bold"), relief="flat")
        sty.map("R.Treeview", background=[("selected", "#BBDEFB")],
                              foreground=[("selected", "#0D47A1")])
        vsb  = ttk.Scrollbar(tf, orient=tk.VERTICAL)
        tree = ttk.Treeview(tf, show="headings", style="R.Treeview",
                            yscrollcommand=vsb.set)
        vsb.config(command=tree.yview); vsb.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True)
        tree.tag_configure("odd",  background="#FFFFFF")
        tree.tag_configure("even", background="#F0F4FF")

        ft2 = tk.Frame(inner, bg="#E3F2FD", height=38)
        ft2.pack(fill=tk.X, side=tk.BOTTOM); ft2.pack_propagate(False)
        sl = tk.Label(ft2, text="", font=("Arial", 10, "bold"),
                      fg="#1565C0", bg="#E3F2FD")
        sl.pack(side=tk.LEFT, padx=12, pady=8)

        VC  = "Processing Time"
        st  = {"dc": [], "ac": [], "raw": [], "disp": []}
        CW  = {"datetime": 145, "travlot": 150, "nrinspector": 120,
               "glassinput": 80, "ok": 60, "ng": 60, "other": 70,
               "starttime": 130, "endtime": 130, VC: 115}
        ss: Dict = {}

        def _pt(s, e):
            try:
                if not s or not e: return ""
                fmt = "%Y-%m-%d %H:%M:%S"
                ts  = s if isinstance(s, dt) else dt.strptime(str(s), fmt)
                te  = e if isinstance(e, dt) else dt.strptime(str(e), fmt)
                d   = int((te - ts).total_seconds())
                if d < 0: return ""
                h, r = divmod(d, 3600); m2, s2 = divmod(r, 60)
                return f"{h:02d}:{m2:02d}:{s2:02d}"
            except: return ""

        def _sc(cols):
            st["dc"] = list(cols); st["ac"] = list(cols) + [VC]
            tree.config(columns=st["ac"])
            for c in st["ac"]:
                tree.heading(c, text=c.upper(),
                             command=lambda cc=c: _sort(cc, False))
                tree.column(c, width=CW.get(c, 100), minwidth=50, anchor="center")

        def _td(raw):
            v  = [x if x is not None else "" for x in raw]
            dc = st["dc"]
            si = dc.index("starttime") if "starttime" in dc else -1
            ei = dc.index("endtime")   if "endtime"   in dc else -1
            v.append(_pt(v[si] if si >= 0 else "", v[ei] if ei >= 0 else ""))
            return v

        def _load(disp):
            tree.delete(*tree.get_children())
            ac  = st["ac"]; oi = ac.index("ok") if "ok" in ac else -1
            tot = 0
            for i, row in enumerate(disp):
                tree.insert("", "end", values=row,
                            tags=("even" if i % 2 == 0 else "odd",))
                try: tot += int(row[oi] or 0) if oi >= 0 else 0
                except: pass
            sl.config(text=f"  {len(disp)} รายการ   |   ✅ Placed รวม: {tot}")

        def _filter(*_):
            kw = sv.get().strip().lower()
            st["disp"] = (
                [_td(r) for r in st["raw"] if any(kw in str(v).lower() for v in r)]
                if kw else [_td(r) for r in st["raw"]])
            _load(st["disp"])

        def _sort(col, rev):
            if not st["ac"]: return
            ci = st["ac"].index(col)
            try:
                st["disp"].sort(
                    key=lambda r: (r[ci] == "" or r[ci] is None, r[ci]), reverse=rev)
            except:
                st["disp"].sort(key=lambda r: str(r[ci] or ""), reverse=rev)
            _load(st["disp"]); ss[col] = not rev
            for c in st["ac"]:
                tree.heading(c, text=c.upper(),
                             command=lambda cc=c: _sort(cc, ss.get(cc, False)))
            tree.heading(col, text=col.upper() + (" ▲" if not rev else " ▼"),
                         command=lambda: _sort(col, ss.get(col, False)))

        def _fetch():
            try:
                df  = dt.strptime(fv.get(), "%Y-%m-%d")
                dt2 = dt.strptime(tv.get(), "%Y-%m-%d").replace(
                    hour=23, minute=59, second=59)
            except:
                play_alarm_beep()
                messagebox.showwarning("⚠️", "รูปแบบวันที่ต้องเป็น YYYY-MM-DD",
                                       parent=win)
                return
            c = cx = None
            try:
                c  = self.data_manager.mysql_pool.get_connection()
                cx = c.cursor()
                cx.execute(
                    "SELECT * FROM fp_clvi_pskipnrins "
                    "WHERE datetime BETWEEN %s AND %s ORDER BY datetime DESC",
                    (df.strftime("%Y-%m-%d %H:%M:%S"),
                     dt2.strftime("%Y-%m-%d %H:%M:%S")))
                raw  = cx.fetchall()
                cols = [d[0] for d in cx.description]
                if not st["dc"]: _sc(cols)
                st["raw"] = raw; _filter()
            except Exception as e:
                logger.error(f"report fetch: {e}")
                play_alarm_beep()
                messagebox.showerror("❌", str(e), parent=win)
            finally:
                if cx: cx.close()
                if c:  c.close()

        def _export():
            if not st["disp"]:
                messagebox.showinfo("ℹ️", "ไม่มีข้อมูล", parent=win); return
            fp = filedialog.asksaveasfilename(
                parent=win, defaultextension=".csv",
                filetypes=[("CSV", "*.csv")],
                initialfile=f"PlaceReport{dt.now().strftime('%Y%m%d_%H%M%S')}.csv")
            if not fp: return
            try:
                with open(fp, "w", newline="", encoding="utf-8-sig") as f:
                    w = csv.writer(f)
                    w.writerow(st["ac"]); w.writerows(st["disp"])
                messagebox.showinfo("✅ Export สำเร็จ", fp, parent=win)
            except Exception as e:
                play_alarm_beep()
                messagebox.showerror("❌", str(e), parent=win)

        rb.config(command=_fetch); eb.config(command=_export)
        sv.trace_add("write", _filter)
        fv.trace_add("write", lambda *_: _fetch())
        tv.trace_add("write", lambda *_: _fetch())
        _fetch(); se.focus()


# ==================== ENTRY POINT ====================
def main():
    logger.info("Smart Place Monitoring System Ver.1.0 starting...")
    try:
        root = tk.Tk()
        app  = DetectionApp(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.error(f"Fatal: {e}")
    finally:
        logger.info("Terminated")


if __name__ == "__main__":
    main()

