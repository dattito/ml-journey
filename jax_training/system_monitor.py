import psutil
import time
import threading
import jax

try:
    import pynvml

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class SystemMonitor:
    def __init__(self, writer, interval=1.0):
        self.writer = writer
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.start_time = time.time()
        self.nvml_initialized = False

        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.nvml_initialized = True
            except Exception as e:
                print(f"Warning: Could not initialize NVML: {e}")

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join()

        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _monitor_loop(self):
        step = 0
        while not self.stop_event.is_set():
            cpu_percent = psutil.cpu_percent()
            ram = psutil.virtual_memory()
            self.writer.add_scalar("System/CPU_Usage_Percent", cpu_percent, step)
            self.writer.add_scalar("System/RAM_Usage_Percent", ram.percent, step)

            if self.nvml_initialized:
                try:
                    num_gpus = pynvml.nvmlDeviceGetCount()
                    for i in range(num_gpus):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        
                        self.writer.add_scalar(f"GPU/{i}/Utilization_Percent", util.gpu, step)
                        self.writer.add_scalar(f"GPU/{i}/Memory_Used_GB", mem.used / 1e9, step)
                except pynvml.NVMLError:
                    pass

            try:
                for i, device in enumerate(jax.local_devices()):
                    mem_stats = device.memory_stats()
                    if mem_stats:
                        used = mem_stats.get("bytes_in_use", 0)
                        self.writer.add_scalar(f"JAX/{i}/Mem_Used_GB", used / 1e9, step)
            except Exception:
                pass

            step += 1
            time.sleep(self.interval)
