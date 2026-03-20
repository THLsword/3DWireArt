from mimetypes import init
import os

def find_index_with_tolerance(lst, target, tol=1e-6):
    """
    在 lst (list of [x,y,z]) 中查找与 target 近似相等的坐标索引，允许一定浮点误差。
    如果找不到，返回 -1
    """
    for i, item in enumerate(lst):
        if (abs(item[0] - target[0]) < tol and
            abs(item[1] - target[1]) < tol and
            abs(item[2] - target[2]) < tol):
            return i
    return -1
    
dir = "data/templates/cup24"
vertices_path = os.path.join(dir, "vertices.txt")

processed_vertices = []
init_params = []
for i, l in enumerate(open(vertices_path, 'r')):
    value = l.strip().split(' ')
    if value[0] == 'RegularVertex':
        _, a, b, c = value
        init_params.append([float(a), float(b), float(c)])
        processed_vertices.append(i)

f = open(os.path.join(dir, 'symmetries.txt'), 'w')
processed = []
# for idx in processed_vertices:
#     if idx in processed:
#         continue
#     vertice = init_params[idx]
#     idx2 = init_params.index([-vertice[0], vertice[1], vertice[2]])
#     if idx == idx2:
#         continue
#     processed.append(idx)
#     processed.append(idx2)
#     f.write(f"{idx} {idx2}\n")

# new
for idx in processed_vertices:
    if idx in processed:
        continue
    vertice = init_params[idx]
    mirror = [-vertice[0], vertice[1], vertice[2]]
    idx2 = find_index_with_tolerance(init_params, mirror, tol=1e-2)
    
    if idx2 == -1:
        # 说明没有找到对应镜像点，可能跳过或打印提示
        print(f"没有在 init_params 中找到镜像顶点：{mirror}")
        continue
    
    # 如果找到镜像点，再继续你的后续逻辑
    processed.append(idx)
    processed.append(idx2)
    f.write(f"{idx} {idx2}\n")