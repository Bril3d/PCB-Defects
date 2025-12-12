from roboflow import Roboflow
rf = Roboflow(api_key="V9M0ITdvfiCCXdCxPVib")
project = rf.workspace("university-2xdiy").project("pcb-defects-chi1b")
version = project.version(6)
dataset = version.download("yolov8")
print("✅ Dataset downloaded!")