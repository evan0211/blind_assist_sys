# 檔名：realtime_ultrasonic_plot.py
import sys, time, math
from collections import deque
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

import serial
import serial.tools.list_ports as list_ports

# ====== 可調參數 ======
TARGET_FPS        = 60           # 繪圖更新率
SAMPLE_LIMIT_SEC  = 20           # 顯示最近N秒
YMAX_DEFAULT      = 500          # 初始Y上限（cm）
EMA_ALPHA         = 0.3        # EMA 平滑強度
MEDIAN_WIN        = 7          # 中位數窗口(奇數>=3)
PORT              = "COM14"       # 例如 "COM6"，None 則自動偵測第一個
BAUD              = 115200
# ======================

def auto_select_port():
    if PORT: return PORT
    ports = list(list_ports.comports())
    if not ports:
        raise RuntimeError("找不到任何序列埠，請先接上 Arduino。")
    prefer = [p for p in ports if any(k in (p.description or '').lower()
                                      for k in ['arduino', 'wch', 'ch340', 'usb'])]
    return (prefer[0] if prefer else ports[0]).device

def median_last(x, win):
    if len(x) < win: return x[-1]
    return float(np.median(x[-win:]))

class RealtimePlot:
    def __init__(self):
        port = auto_select_port()
        self.ser = serial.Serial(port, BAUD, timeout=0.01)
        time.sleep(1.8)  # 等 Arduino 重置
        print(f"[Serial] {port} @ {BAUD}")

        maxlen = TARGET_FPS * SAMPLE_LIMIT_SEC * 2
        self.ts  = deque(maxlen=maxlen)
        self.raw = deque(maxlen=maxlen)
        self.smf = deque(maxlen=maxlen)
        self._t0 = time.time()
        self._ema = None

        self.app = QtWidgets.QApplication(sys.argv)
        pg.setConfigOptions(antialias=True)
        self.win = pg.GraphicsLayoutWidget(show=True, title="HC-SR04 即時距離（平滑不卡頓）")
        self.win.resize(900, 500)

        self.plot = self.win.addPlot()
        self.plot.setLabel('left',  'Distance', units='cm')
        self.plot.setLabel('bottom','Time',     units='s')
        self.plot.setYRange(0, YMAX_DEFAULT)
        self.plot.showGrid(x=True, y=True, alpha=0.3)

        self.curve_raw = self.plot.plot(pen=pg.mkPen(width=1))
        self.curve_smf = self.plot.plot(pen=pg.mkPen(width=2))

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(int(1000 / TARGET_FPS))

        self.x_timer = QtCore.QTimer()
        self.x_timer.timeout.connect(self.update_view)
        self.x_timer.start(500)

    def close(self):
        try: self.ser.close()
        except: pass

    def update(self):
        try:
            while self.ser.in_waiting:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if not line: continue
                try:
                    d = float(line)
                    if not math.isfinite(d) or d <= 0: continue
                except:
                    continue

                t = time.time() - self._t0
                self.ts.append(t)
                self.raw.append(d)

                med = median_last(self.raw, MEDIAN_WIN)
                self._ema = med if self._ema is None else (EMA_ALPHA * med + (1.0 - EMA_ALPHA) * self._ema)
                self.smf.append(self._ema)

            if self.ts:
                t_now = self.ts[-1]
                t_min = max(0.0, t_now - SAMPLE_LIMIT_SEC)

                ts  = np.array(self.ts)
                raw = np.array(self.raw)
                smf = np.array(self.smf)
                m = ts >= t_min
                self.curve_raw.setData(ts[m], raw[m])
                self.curve_smf.setData(ts[m], smf[m])

        except Exception as e:
            sys.stdout.write(f"\r[Warn] {e}  ")
            sys.stdout.flush()

    def update_view(self):
        if not self.ts: return
        t_now = self.ts[-1]
        self.plot.setXRange(max(0.0, t_now - SAMPLE_LIMIT_SEC), max(SAMPLE_LIMIT_SEC, t_now), padding=0.02)

        if self.smf:
            ymax = float(np.max(self.smf))
            target = max(500, (int(ymax / 50) + 1) * 50)  # 上限至少 800
            y0, y1 = self.plot.viewRange()[1]
            if target - y1 > 10 or target < y1 * 0.6:
                self.plot.setYRange(0, target, padding=0.02)

    def run(self):
        ret = self.app.exec_()
        self.close()
        return ret

if __name__ == "__main__":
    try:
        app = RealtimePlot()
        sys.exit(app.run())
    except Exception as e:
        print(f"[Fatal] {e}")
        sys.exit(1)
