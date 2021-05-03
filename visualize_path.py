import sys

# Code from - https://github.com/StanfordVL/GibsonEnv/blob/30b7e7c1f352ffadc67f60abdf3bdcb2fe52f7d7/gibson/data/visualize_path.py
import bpy
import argparse
import numpy as np
import json
from mathutils import Matrix, Vector, Euler
import json
import random

from os import path as osp

def import_obj(file_loc):
    imported_object = bpy.ops.import_scene.obj(filepath=file_loc)
    obj_object = bpy.context.selected_objects[0] ####<--Fix
    print('Imported name: ', obj_object.name)
    model = bpy.context.object
    return model


def line_distance(pt1, pt2):
    return np.linalg.norm(pt1 - pt2)

def visualize(path):
    d1 = makeMaterial('Orange',(1, 0.25, 0, 1),(1,1,1),1)
    d2 = makeMaterial('Green',(0, 1, 0, 1),(1,1,1),1)
    d3 = makeMaterial('Blue',(0, 0, 1, 1),(1,1,1),1)
    
    start = path[0]
    end = path[-1]

    start_loc = [start[0], start[1], start[2]]
    end_loc = [end[0], end[1], end[2]]
    end_obj = bpy.ops.mesh.primitive_uv_sphere_add(location=end_loc, radius=0.2)
    setMaterial("Sphere", d2)
    start_obj = bpy.ops.mesh.primitive_uv_sphere_add(location=start_loc, radius=0.1)
    '''setMaterial("Sphere", d2)
    setMaterial("Sphere", d3)'''
    # for loc in path[1:-1]:
    #     y_up_loc = [loc[0], loc[1], loc[2]]
    #     bpy.ops.mesh.primitive_uv_sphere_add(location=y_up_loc, radius=0.05)
    setMaterial("Sphere", d1)

def visualize_traj(traj):
  '''
  Plots the entire trajectory
  :param traj: The trajectory to plot
  '''
  d3 = makeMaterial('Blue',(0,0,1, 1),(1,1,1),1)

  for loc in traj[1:-1]:
    y_up_loc = [loc[0], loc[1], loc[2]]
    bpy.ops.mesh.primitive_uv_sphere_add(location=y_up_loc, radius=0.05)
  setMaterial("Sphere", d3)


def visualize_curve(traj, color):
  '''
  Visualize the trajectory using path curve
  :param traj: The trajectory to plot
  :parma color: The color of the curve
  '''
  crv = bpy.data.curves.new('crv', 'CURVE')
  crv.dimensions = '3D'
  crv.bevel_depth=0.01

  spline = crv.splines.new(type='NURBS')
  spline.points.add(len(traj)-1)

  for p, new_co in zip(spline.points, traj):
    p.co = (new_co + [1.0])

  # Setting the color
  material = bpy.data.materials.new('crv_material')
  material.diffuse_color = color
  crv.materials.append(material)

  obj = bpy.data.objects.new('trajectory', crv)
  bpy.context.scene.collection.objects.link(obj)


def makeMaterial(name, diffuse, specular, alpha):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = diffuse
    # mat.diffuse_shader = 'LAMBERT'
    # mat.diffuse_intensity = 1.0
    # mat.specular_color = specular
    # mat.specular_shader = 'COOKTORR'
    # mat.specular_intensity = 0.5
    # mat.alpha = alpha
    # mat.ambient = 1
    return mat

def setMaterial(name, mat):
    for ob in bpy.data.objects:
        if name in ob.name:
            me = ob.data
            me.materials.append(mat)
            ob.name = "PathPoint"


def duplicateObject(scene, name, copyobj):
    # Create new mesh
    mesh = bpy.data.meshes.new(name)
    # Create new object associated with the mesh
    ob_new = bpy.data.objects.new(name, mesh)
    # Copy data block from the old object into the new object
    ob_new.data = copyobj.data.copy()
    ob_new.scale = copyobj.scale
    ob_new.location = copyobj.location
    # Link new object to the given scene and select it
    scene.collection.objects.link(ob_new)
    ob_new.select_set(True)
    ob_new.location = Vector((0, 0, 0))
    return ob_new

def moveFromCenter(obj, dx=2000, dy=2000, dz=2000):
  obj.location = Vector((dx, dy, dz))


def prepare():
  bpy.ops.object.select_all(action='DESELECT')
  if 'Cube' in bpy.data.objects.keys():
    mesh = bpy.data.meshes["Cube"]
    bpy.data.meshes.remove(mesh)
  bpy.ops.object.delete() 
  # lamp = bpy.data.lights.new(name='Lamp', type='SUN')
  lamp = bpy.data.lights['Light']
  lamp.energy = 1.0  # 10 is the max value for energy
  lamp.type = 'SUN'  # in ['POINT', 'SUN', 'SPOT', 'HEMI', 'AREA']
  lamp.distance = 100

def install_lamp(obj_lamp, loc_lamp, loc_target):
  direction = loc_target - loc_lamp
  rot_quat = direction.to_track_quat('-Z', 'Y')
  mat_loc = Matrix.Translation(loc_lamp)
  mat_rot = rot_quat.to_matrix().to_4x4()
  mat_comb = mat_loc * mat_rot
  obj_lamp.matrix_world = mat_comb

def look_at(obj_camera, loc_camera, loc_target):
  '''Set camera to look at loc_target from loc_camera
  Camera default y is up
  '''
  direction = loc_target - loc_camera
  rot_quat = direction.to_track_quat('-Z', 'Y')
  mat_loc = Matrix.Translation(loc_camera)
  mat_rot = rot_quat.to_matrix().to_4x4()
  mat_comb = mat_loc @ mat_rot
  obj_camera.matrix_world = mat_comb


def get_model_camera_vals(filepath):
  all_x, all_y, all_z = [], [], []
  with open(filepath, "r") as f:
    for line in f:
      vals = line.split(",")
      all_x.append(float(vals[1]))
      all_y.append(float(vals[2]))
      all_z.append(float(vals[3]))
  max_x, min_x = (max(all_x), min(all_x))
  max_y, min_y = (max(all_y), min(all_y))
  max_z, min_z = (max(all_z), min(all_z))
  center = Vector(((max_x + min_x)/2, (max_y + min_y)/2, (max_z + min_z)/2))
  return (max_x, min_x), (max_y, min_y), (max_z, min_z), center 

def join_objects():
    scene = bpy.context.scene
    obs = []
    for ob in scene.objects:
        if ob.type == 'MESH':
            obs.append(ob)
    ctx = bpy.context.copy()
    ctx['active_object'] = obs[0]
    ctx['selected_objects'] = obs
    ctx['selected_editable_bases'] = obs #[scene.object_bases[ob.name] for ob in obs]
    bpy.ops.object.join(ctx)


def deleteObject(obj):
  '''for ob in bpy.data.objects:
    print(ob)
    ob.select = False'''
  bpy.ops.object.mode_set(mode='OBJECT')
  bpy.ops.object.select_all(action='DESELECT')
  if type(obj) == str:
      bpy.data.objects[obj].select_set(True)
  else:
      obj.select_set(True)
  for name in bpy.data.objects.keys():
    if "PathPoint" in name:
      bpy.data.objects[name].select_set(True)
  bpy.ops.object.delete() 

def deleteSpheres():
  bpy.ops.object.mode_set(mode='OBJECT')
  bpy.ops.object.select_all(action='DESELECT')
  for name in bpy.data.objects.keys():
    if "Sphere" in name:
      bpy.data.objects[name].select_set(True)
  bpy.ops.object.delete() 


def deleteCube():
  for name, obj in bpy.data.objects.items():
      if "Cube" in name:
          bpy.data.objects.remove(obj, True)


def capture_top(dst_dir, model_id, obj_model, focus_center, path, distance, exp):
  def set_render_resolution(x=2560, y=2560):
    bpy.context.scene.render.resolution_x = x
    bpy.context.scene.render.resolution_y = y
  set_render_resolution()
  camera_pos = focus_center + Vector((0, 0, distance))
  lamp_pos = camera_pos
  obj_camera = bpy.data.objects["Camera"]
  obj_camera.location = camera_pos
  obj_lamp = bpy.data.objects["Light"]
  obj_lamp.location = camera_pos
  look_at(obj_camera, camera_pos, focus_center)
  install_lamp(obj_lamp, lamp_pos, focus_center)
  slicename="slice"
  cut_height = np.mean([loc[2] for loc in path])
  cobj=duplicateObject(bpy.context.scene, slicename, obj_model)
  bpy.ops.object.select_all(action='DESELECT')
  bpy.context.view_layer.objects.active = bpy.data.objects[slicename]
  bpy.ops.object.mode_set(mode='EDIT')
  bpy.ops.mesh.select_all(action='SELECT')
  bpy.ops.mesh.bisect(plane_co=(0, 0, cut_height + 0.7),plane_no=(0,0,1), clear_outer=True,clear_inner=False)
  bpy.ops.object.mode_set(mode='OBJECT')
  bpy.ops.object.select_all(action='DESELECT')
  bpy.data.scenes['Scene'].render.filepath = osp.join(dst_dir, 'compare_path.png')
  bpy.ops.render.render( write_still=True, scene='Scene') 
  # deleteObject(slicename)

# def parse_local_args( args ):
#   local_args = args[ args.index( '--' ) + 1: ]
#   return parser.parse_known_args( local_args )

# parser = argparse.ArgumentParser()
# parser.add_argument('--filepath', required=True, help='trajectory file path', type=str)
# parser.add_argument('--datapath', required=True, help='gibson dataset path', type=str)
# parser.add_argument('--renderpath', help='visualization output path', default=None, type=str)
# parser.add_argument('--model', required=True, type=str)
# parser.add_argument('--idx'  , default=0, type=int)

import pickle
def main():
  # global args, logger 
  # opt, remaining_args = parse_local_args( sys.argv )

  trajectories = {}
  # json_path = os.path.join(opt.filepath, "{}.json".format(opt.model))
  # with open(json_path, "r") as f:
  #     trajectories = json.load(f)
  exp = 'rrt_star'
  path_param = pickle.load(open('/home/jacoblab/prob_planning_data/gibson_path_{}.p'.format(exp), 'rb'))
  waypoints = [[p[0], p[1], 0.5] for p in path_param['path']]
  traj_rrt = [[p[0], p[1], 0.5] for p in path_param['path_interpolated']]
  exp = 'ccgp'
  path_param = pickle.load(open('/home/jacoblab/prob_planning_data/gibson_path_{}_5.p'.format(exp), 'rb'))
  traj_ccgp = [[p[0], p[1], 0.5] for p in path_param['path_interpolated']]


  prepare()
  datapath = '/home/jacoblab/prob_planning/assets'
  model = 'Allensville'
  import_obj(osp.join(datapath, model, "mesh_z_up.obj"))
  camera_pose = osp.join(datapath, model, "camera_poses.csv")

  join_objects()
  obj_model, cobj = bpy.data.objects[2], None
  moveFromCenter(obj_model)
  (max_x, min_x), (max_y, min_y), (max_z, min_z), _ = get_model_camera_vals(camera_pose)
  dist = max(((max_x - min_x), (max_y - min_y), (max_z - min_z))) / (2*np.tan(np.pi/10))
  cent = Vector(((max_x + min_x)/2, (max_y + min_y)/2, (max_z + min_z)/2))

  renderpath = datapath #if opt.renderpath is None else opt.renderpath
  visualize(waypoints)
  visualize_curve(traj_rrt, (1, 0, 0, 1))
  visualize_curve(traj_ccgp, (0, 0, 1, 1))

  capture_top(renderpath, model, obj_model, cent, waypoints, dist, exp)
        
if __name__ == '__main__':
  main()