import mitsuba as mi
import drjit as dr
mi.set_variant('cuda_ad_rgb')
import matplotlib.pyplot as plt

scene = mi.load_file('./scenes/cbox.xml')

cam_origin = mi.Point3f(0,1,3)
cam_dir = dr.normalize(mi.Vector3f(0,-0.5,-1))
cam_width = 2.0
cam_height = 2.0
image_res=(256,256)

x, y = dr.meshgrid(
    dr.linspace(mi.Float, -cam_width  / 2,   cam_width / 2, image_res[0]),
    dr.linspace(mi.Float, -cam_height / 2,  cam_height / 2, image_res[1])
)
ray_origin_local = mi.Vector3f(x, y, 0)
ray_origin = mi.Frame3f(cam_dir).to_world(ray_origin_local) + cam_origin

ray = mi.Ray3f(o=ray_origin, d = cam_dir)
si = scene.ray_intersect(ray)

ambient_range = 0.75
ambient_ray_count = 256
rng = mi.PCG32(size=dr.prod(image_res))

result = mi.Float(0)

@dr.syntax
def my_loop(result,si,rng,ambient_ray_count):
    i = mi.UInt32(0)
    while(si.is_valid() & (i < ambient_ray_count)):
        sample_1, sample_2 = rng.next_float32(),rng.next_float32()
        wo_local = mi.warp.square_to_uniform_hemisphere([sample_1, sample_2])
        wo_world = si.sh_frame.to_world(wo_local)
        ray_2 = si.spawn_ray(wo_world)
        ray_2.maxt = ambient_range
        result[~scene.ray_test(ray_2)] += 1.0
        i += 1
    return result / ambient_ray_count

result = my_loop(result, si, rng, ambient_ray_count)

image = mi.TensorXf(result,shape=image_res)
plt.imshow(image, cmap='gray'); plt.axis('off');
plt.savefig('./images/result5.png')