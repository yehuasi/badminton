#!/bin/sh
# SC171 v3 板子端启动脚本：进入部署目录并启动 Web 服务（日志追加到 server.log）。
#
# 启动（在 PC 上，经 start-stop-daemon 守护，可脱离 adb 会话存活）：
#   adb push scripts\start_board.sh /data/local/tmp/badminton/
#   adb shell 'start-stop-daemon -S -b -m -p /data/local/tmp/badminton/server.pid -d /data/local/tmp/badminton -x /data/local/tmp/badminton/start_board.sh'
# 可选第 1 个参数指定姿态模型（默认 yolov8n-pose-480.onnx，追求速度可传 yolov8n-pose-416.onnx）：
#   ... -x /data/local/tmp/badminton/start_board.sh -- yolov8n-pose-416.onnx
#
# 查看：tail -f /data/local/tmp/badminton/server.log
# 停止：adb shell 'start-stop-daemon -K -p /data/local/tmp/badminton/server.pid'
#
# 摄像头设备 /dev/video2 对应 --source 2；如需演示模式把下面 --source 2 改成 --source demo。
POSE=${1:-yolov8n-pose-480.onnx}
cd /data/local/tmp/badminton || exit 1
exec python3 -m app.server --backend onnx --source 2 --host 0.0.0.0 --port 8000 --pose-model "$POSE" >> /data/local/tmp/badminton/server.log 2>&1
