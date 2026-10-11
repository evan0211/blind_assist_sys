import argparse, time, sys, cv2, numpy as np
from ultralytics import YOLO

def args_():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="YOLO11 .pt")
    ap.add_argument("-s", "--source", default="0", help="攝影機索引(0/1...)或RTSP/影片路徑")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--save", default="", help="輸出 mp4 檔(選填)")
    ap.add_argument("--device", default="", help="'0'或'cpu'；空字串自動")
    return ap.parse_args()

def palette(n):
    rng = np.random.default_rng(0)
    return [tuple(int(x) for x in rng.integers(64,255,3)) for _ in range(n or 1)]

def draw(frame, dets, names):
    n_cls = max((int(b["cls"]) for b in dets), default=-1) + 1
    colors = palette(n_cls)
    for b in dets:
        x1,y1,x2,y2 = b["xyxy"]; c=int(b["cls"]); p=float(b["conf"])
        color = colors[c % len(colors)]
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        label = f"{names.get(c,str(c))} {p:.2f}"
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame,(x1,y1-th-6),(x1+tw+6,y1),color,-1)
        cv2.putText(frame,label,(x1+3,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2,cv2.LINE_AA)
    return frame

def main():
    args = args_()
    model = YOLO(args.model)  # 這裡直接吃 YOLO11 的 .pt
    names = model.model.names if hasattr(model, "model") else model.names

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened(): print("無法開啟來源:", args.source); sys.exit(1)

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w,h))
        print(f"[Save] {args.save} ({w}x{h}@{fps:.1f})")

    t0=time.time(); n=0
    while True:
        ok, frame = cap.read()
        if not ok: break
        n+=1

        # YOLO11 推論（單張 frame）
        results = model.predict(
            frame, imgsz=args.imgsz, conf=0.5,
            device=args.device if args.device else None, verbose=False
        )
        r = results[0]
        dets=[]
        if r.boxes is not None and len(r.boxes):
            for b in r.boxes:
                x1,y1,x2,y2 = b.xyxy[0].int().tolist()
                cls = int(b.cls[0].item()) if b.cls is not None else 0
                conf = float(b.conf[0].item()) if b.conf is not None else 0.0
                dets.append({"xyxy":(x1,y1,x2,y2),"cls":cls,"conf":conf})

        out = draw(frame.copy(), dets, names)

        fps = n / (time.time()-t0+1e-6)
        cv2.putText(out, f"FPS:{fps:.1f}", (10,24), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2,cv2.LINE_AA)

        if writer: writer.write(out)
        cv2.imshow("YOLO11 Detect", out)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')): break

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
