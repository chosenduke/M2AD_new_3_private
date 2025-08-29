import torch, re

path = "/root/autodl-tmp/M2AD/backbones/weights/dinov2_vitb14_reg4_pretrain.pth"
obj = torch.load(path, map_location="cpu")

sd = None
for k in ["model", "state_dict", "student", "teacher", "module"]:
    if isinstance(obj, dict) and k in obj:
        sd = obj[k]
        break
if sd is None:
    sd = obj if isinstance(obj, dict) else {}

pat = re.compile(r"(?:blocks|block|layers|layer|encoder\.layers|transformer\.blocks)\.(\d+)\.")
indices = sorted({int(m.group(1)) for name in sd.keys() for m in [pat.search(name)] if m})

print("检测到的层数:", len(indices))
print("样例索引:", indices[:3], "...", indices[-3:])