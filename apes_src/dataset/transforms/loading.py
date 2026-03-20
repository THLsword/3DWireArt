from .basetransform import BaseTransform
from typing import Dict
import numpy as np
from PIL import Image

class LoadPCD(BaseTransform):
    def transform(self, results: Dict) -> Dict:
        pcd_path = results['pcd_path']
        pcd = np.load(pcd_path)
        # 将点云平移到原点
        centroid = np.mean(pcd, axis=0)
        point_cloud_centered = pcd - centroid
        # 缩放点云使其适应[-1, 1]范围
        max_distance = np.max(np.sqrt(np.sum(point_cloud_centered ** 2, axis=1)))
        point_cloud_normalized = point_cloud_centered / max_distance
        results['pcd'] = point_cloud_normalized
        return results

class LoadCLSLabel(BaseTransform):
    def transform(self, results: Dict) -> Dict:
        cls_label_path = results['cls_label_path']
        label = np.load(cls_label_path)
        results['cls_label'] = label
        return results

class LoadSEGLabel(BaseTransform):
    def transform(self, results: Dict) -> Dict:
        seg_label_path = results['seg_label_path']
        label = np.load(seg_label_path)
        results['seg_label'] = label
        return results

class LoadMultiview(BaseTransform):
    def transform(self, results: Dict) -> Dict:
        multi_view_paths = results['multi_view_paths'] # list
        multi_view_imgs = [Image.open(path) for path in multi_view_paths]
        np_imgs = [np.array(img) for img in multi_view_imgs]
        normalized_np_imgs = [img.astype(np.float32) / 255 for img in np_imgs]
        results['multi_view_imgs'] = normalized_np_imgs
        return results

class LoadRender(BaseTransform):
    def transform(self, results: Dict) -> Dict:
        render_paths = results['render_paths'] # list
        render_imgs = [Image.open(path).convert('L') for path in render_paths]
        np_imgs = [np.array(img) for img in render_imgs]
        normalized_np_imgs = [img.astype(np.float32) / 255 for img in np_imgs]
        results['render_imgs'] = normalized_np_imgs
        return results