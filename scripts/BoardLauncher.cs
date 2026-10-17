using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Net;
using System.Text;
using System.Windows.Forms;

namespace BadmintonBoard
{
    public class LauncherForm : Form
    {
        const string BoardDir = "/data/local/tmp/badminton";
        const string PidFile = BoardDir + "/server.pid";
        const string StartSh = BoardDir + "/start_board.sh";
        const string LocalUrl = "http://127.0.0.1:18000";

        Button btnStart;
        Button btnStop;
        TextBox txtStatus;

        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new LauncherForm());
        }

        public LauncherForm()
        {
            Text = "羽毛球端侧系统 · 一键启动/关闭";
            ClientSize = new Size(470, 320);
            StartPosition = FormStartPosition.CenterScreen;
            FormBorderStyle = FormBorderStyle.FixedSingle;
            MaximizeBox = false;

            var lbl = new Label
            {
                Text = "开发板控制",
                Font = new Font("Microsoft YaHei", 12f, FontStyle.Bold),
                Location = new Point(20, 16),
                AutoSize = true,
            };

            btnStart = new Button
            {
                Text = "打开程序（启动服务 + 打开网页）",
                Location = new Point(20, 58),
                Size = new Size(420, 48),
                Font = new Font("Microsoft YaHei", 10f),
            };
            btnStart.Click += (s, e) => RunAsync(StartFlow);

            btnStop = new Button
            {
                Text = "关闭程序（停止服务 + 断开转发）",
                Location = new Point(20, 116),
                Size = new Size(420, 48),
                Font = new Font("Microsoft YaHei", 10f),
            };
            btnStop.Click += (s, e) => RunAsync(StopFlow);

            txtStatus = new TextBox
            {
                Location = new Point(20, 178),
                Size = new Size(420, 120),
                Multiline = true,
                ReadOnly = true,
                ScrollBars = ScrollBars.Vertical,
                Font = new Font("Microsoft YaHei", 9f),
            };

            Controls.Add(lbl);
            Controls.Add(btnStart);
            Controls.Add(btnStop);
            Controls.Add(txtStatus);

            Shown += (s, e) =>
            {
                if (!AdbAvailable())
                {
                    Append("未找到 adb（请确认 adb 在 PATH 中）。");
                    return;
                }
                Append(GetBoardSerial() != null
                    ? "开发板已连接，可以点击「打开程序」。"
                    : "未检测到开发板，请连接数据线后重试。");
            };
        }

        void RunAsync(Action flow)
        {
            btnStart.Enabled = false;
            btnStop.Enabled = false;
            var t = new System.Threading.Thread(() =>
            {
                try { flow(); }
                catch (Exception ex) { Append("异常：" + ex.Message); }
                finally
                {
                    Invoke((Action)(() =>
                    {
                        btnStart.Enabled = true;
                        btnStop.Enabled = true;
                    }));
                }
            });
            t.IsBackground = true;
            t.Start();
        }

        void StartFlow()
        {
            Append(TimeNow() + " 检查 adb 与开发板连接…");
            if (!AdbAvailable())
            {
                Append("未找到 adb，请先安装并加入 PATH。");
                return;
            }
            if (GetBoardSerial() == null)
            {
                Append("未检测到开发板（device 状态）。请用数据线连接后重试。");
                return;
            }

            Append("检测到开发板，开始启动（约 15~25 秒）…");
            RunAdb("root", 20000);
            SleepSeconds(2);
            RunAdb("forward --remove-all", 5000);
            RunAdb("forward tcp:18000 tcp:8000", 5000);

            RunAdb("shell start-stop-daemon -K -p " + PidFile + " 2>/dev/null", 5000);
            RunAdb("shell pkill -f app.server 2>/dev/null", 5000);
            SleepSeconds(1);
            RunAdb("shell rm -f " + PidFile, 5000);

            RunAdb("shell start-stop-daemon -S -b -m -p " + PidFile + " -d " + BoardDir + " -x " + StartSh, 10000);

            bool ready = false;
            for (int i = 0; i < 60; i++)
            {
                SleepSeconds(0.5);
                if (UrlReachable(LocalUrl + "/api/status", 2000))
                {
                    ready = true;
                    break;
                }
            }

            if (ready)
            {
                Append("启动成功，正在打开浏览器：" + LocalUrl);
                try
                {
                    Process.Start(new ProcessStartInfo(LocalUrl) { UseShellExecute = true });
                }
                catch (Exception ex)
                {
                    Append("打开浏览器失败：" + ex.Message);
                }
            }
            else
            {
                Append("启动超时：服务未就绪。可手动查看日志：adb shell tail -n 30 " + BoardDir + "/server.log");
            }
        }

        void StopFlow()
        {
            Append(TimeNow() + " 停止服务并断开转发…");
            RunAdb("shell start-stop-daemon -K -p " + PidFile + " 2>/dev/null", 5000);
            RunAdb("shell pkill -f app.server 2>/dev/null", 5000);
            SleepSeconds(1);
            RunAdb("shell rm -f " + PidFile, 5000);
            RunAdb("forward --remove-all", 5000);
            Append("已停止。网页标签页请手动关闭。");
        }

        void Append(string msg)
        {
            if (InvokeRequired)
            {
                Invoke((Action)(() => Append(msg)));
                return;
            }
            txtStatus.AppendText(msg + Environment.NewLine);
            txtStatus.SelectionStart = txtStatus.TextLength;
            txtStatus.ScrollToCaret();
        }

        static string TimeNow()
        {
            return DateTime.Now.ToString("HH:mm:ss");
        }

        static void SleepSeconds(double s)
        {
            System.Threading.Thread.Sleep((int)(s * 1000));
        }

        static string AdbPath()
        {
            var path = Environment.GetEnvironmentVariable("PATH") ?? "";
            foreach (var dir in path.Split(';'))
            {
                var d = dir.Trim();
                if (d.Length == 0) continue;
                var candidate = Path.Combine(d, "adb.exe");
                if (File.Exists(candidate)) return candidate;
            }
            return "adb.exe";
        }

        static string RunAdb(string args, int timeoutMs)
        {
            var psi = new ProcessStartInfo
            {
                FileName = AdbPath(),
                Arguments = args,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8,
            };
            using (var p = Process.Start(psi))
            {
                var stdout = p.StandardOutput.ReadToEnd();
                var stderr = p.StandardError.ReadToEnd();
                if (!p.WaitForExit(timeoutMs))
                {
                    try { p.Kill(); } catch { }
                }
                return stdout + stderr;
            }
        }

        bool AdbAvailable()
        {
            try
            {
                var outTxt = RunAdb("version", 5000);
                return outTxt != null && outTxt.IndexOf("Android Debug Bridge", StringComparison.OrdinalIgnoreCase) >= 0;
            }
            catch
            {
                return false;
            }
        }

        string GetBoardSerial()
        {
            var outTxt = RunAdb("devices", 5000);
            if (outTxt == null) return null;
            foreach (var raw in outTxt.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries))
            {
                var t = raw.Trim();
                if (t.Length == 0 || t.StartsWith("List of devices")) continue;
                var idx = t.IndexOf('\t');
                if (idx < 0) continue;
                var serial = t.Substring(0, idx).Trim();
                var state = t.Substring(idx + 1).Trim();
                if (state == "device") return serial;
            }
            return null;
        }

        static bool UrlReachable(string url, int timeoutMs)
        {
            try
            {
                var req = (HttpWebRequest)WebRequest.Create(url);
                req.Timeout = timeoutMs;
                req.ReadWriteTimeout = timeoutMs;
                using (var resp = (HttpWebResponse)req.GetResponse())
                {
                    return (int)resp.StatusCode == 200;
                }
            }
            catch
            {
                return false;
            }
        }
    }
}
