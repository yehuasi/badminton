# 通过 USB 访问板子 Web 界面（走 adb 端口转发，视频流不经 WiFi，最稳）
# 用法（项目根目录）：powershell -ExecutionPolicy Bypass -File scripts\usb_web.ps1
# 拔插 USB 或 adb 重连后需再执行一次。

adb forward tcp:18000 tcp:8000
if ($LASTEXITCODE -eq 0) {
    Write-Host "已建立 USB 转发。请在浏览器打开: http://127.0.0.1:18000"
} else {
    Write-Host "转发失败，请确认板子已通过 USB 连接（adb devices 能看到设备）。"
}
