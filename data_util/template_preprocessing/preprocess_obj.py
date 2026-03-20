import os
import sys

sys.path.insert(0, os.path.abspath(
	os.path.join(os.path.dirname(__file__), '..')
))

import argparse
import meshio
import numpy as np
import pickle

def load_mesh(path):
    def normalize_mesh(mesh):
        vertices = mesh.points

        v_max = np.amax(vertices, axis = 0)
        v_min = np.amin(vertices, axis = 0)
        v_center = (v_max + v_min) / 2
        vertices -= v_center

        max_dist = np.sqrt(np.max(np.sum(vertices**2, axis=-1)))
        scale = 1. / max_dist
        mesh.points = vertices * scale
        return mesh
    
    mesh = meshio.read(path)
    mesh = normalize_mesh(mesh)
    
    return mesh

def generate_control_points(vertices, faces):
    def calculate_control_points(start, end):
        s = np.array(start)
        e = np.array(end)
        interval = (e - s) / 3
        return s+interval, s+2*interval

    pair_dict = {}
    index_dict = {}
    vertex_list = []
    
    for face in faces:
        for i, index in enumerate(face):
            start_index = index
            end_index = face[i+1] if i < 3 else face[0]
            pair = (start_index, end_index)
            inverse_pair = (end_index, start_index)
            start_vertice = vertices[start_index]
            end_vertice = vertices[end_index]

            # pair check
            if pair in pair_dict.keys() or inverse_pair in pair_dict.keys():
                continue
            
            # start_index check
            if not start_index in index_dict.keys():
                index_dict[start_index] = len(vertex_list)
                vertex_list.append(start_vertice)
            pair_dict[pair] = (len(vertex_list), len(vertex_list)+1)
            pair_dict[inverse_pair] = (len(vertex_list)+1, len(vertex_list))
            v1, v2 = calculate_control_points(start_vertice, end_vertice)
            vertex_list.append(v1)
            vertex_list.append(v2)
            
            # end_index check
            if not end_index in index_dict.keys():
                index_dict[end_index] = len(vertex_list)
                vertex_list.append(end_vertice)
    
    return vertex_list, pair_dict, index_dict
 
def write_vertices(vertex_list, output_dir):
    f = open(os.path.join(output_dir, 'vertices.txt'), 'w')
    for x, y, z in vertex_list:
        x = format(x, '.2f')
        y = format(y, '.2f')
        z = format(z, '.2f')
        f.writelines(f"RegularVertex {x} {y} {z}\n")

def write_topology(faces, pair_dict, index_dict, output_dir):
    topology = []
    for face in faces:
        patch = []
        for i, index in enumerate(face):
            start_index = index
            end_index = face[i+1] if i < 3 else face[0]
            pair = (start_index, end_index)

            patch.append(index_dict[start_index])
            patch.append(pair_dict[pair][0])
            patch.append(pair_dict[pair][1])

        topology.append(patch)

    t = ["["] + [f"{patch},\n" for patch in topology[:len(topology)-1]]
    t = t + [f"{topology[-1]}]\n"]
    with open(os.path.join(output_dir, 'topology.txt'), 'w') as f:
        f.writelines(t)

def write_adjacencies(faces, output_dir):
    f = open(os.path.join(output_dir, 'adjacencies.txt'), 'w')
    faces = np.array(faces)
    for i_face, face in enumerate(faces):
        for i, index in enumerate(face):
            prev_index = face[i-1] if i > 0 else face[-1]
            
            inds = np.where(faces == index)[0]
            for i_patch in inds:
                if i_patch == i_face:
                    continue
                if prev_index in faces[i_patch]:
                    continue
                
                f.write(f"{i_patch} edge {i}, ")
        f.write("\n")

if __name__ == '__main__':
    # 在開始前，先用txt打開obj，把除了v和f的都刪掉(vt,vn)


    parser = argparse.ArgumentParser(description = 'Preprocess template .obj data.')
    parser.add_argument('--data-path', type = str, default = 'data/templates/cup24/cup_template.obj',
					    help = 'Path of template .obj data.')
    parser.add_argument('--output-dir', type = str, default = 'data/templates/cup24',
					    help = 'Directory path of output datas.')          
    args = parser.parse_args()
    
    mesh = load_mesh(args.data_path)
    
    vertices = mesh.points
    faces = mesh.cells_dict['quad']
    print(vertices)
    print("faces: ", faces) 
    for i_face, face in enumerate(faces):
        for i, index in enumerate(face):
            print(index)
    vertex_list, pair_dict, index_dict = generate_control_points(vertices, faces)
    write_vertices(vertex_list, args.output_dir)
    write_topology(faces, pair_dict, index_dict, args.output_dir)
    write_adjacencies(faces, args.output_dir)