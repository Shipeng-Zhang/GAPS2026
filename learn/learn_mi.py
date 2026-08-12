import mitsuba as mi
import matplotlib.pyplot as plt

mi.set_variant('cuda_ad_rgb')

scene = mi.load_file("./scenes/cbox.xml")
image = mi.render(scene, spp=256)

plt.axis("off")
plt.imshow(image ** (1.0/2.2))
plt.savefig("./images/result.png", bbox_inches='tight', pad_inches=0)
mi.util.write_bitmap("./images/my_first_render.png",image)
mi.util.write_bitmap("./images/my_first_render.exr",image)
plt.close()