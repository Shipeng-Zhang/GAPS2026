import mitsuba as mi
import drjit as dr
mi.set_variant('cuda_ad_spectral_polarized')
import matplotlib.pyplot as plt

scene = mi.load_file('./scenes/cbox_pol.xml')
image = mi.render(scene, spp=512)

bitmap = mi.Bitmap(image, channel_names=['R', 'G', 'B'] + scene.integrator().aov_names())
bitmap.write('cbox_pol_output.exr')

channels = dict(bitmap.split())
print(channels.keys())
plt.figure(figsize=(5, 5))
plt.imshow(channels['S0'].convert(srgb_gamma=True), cmap='gray')
plt.colorbar()
plt.xticks([]); plt.yticks([])
plt.xlabel("S0: Intensity", size=14, weight='bold')
plt.savefig('./images/result6.png')

def plot_stokes_component(ax, image):
    data = mi.TensorXf(image)[:, :, 0]
    plot_minmax = 0.05 * max(dr.max(data, axis=None), dr.max(-data, axis=None)).array[0]
    img = ax.imshow(data, cmap='coolwarm', vmin=-plot_minmax, vmax=+plot_minmax)
    ax.set_xticks([]); ax.set_yticks([])
    return img


fig, ax = plt.subplots(ncols=3, figsize=(18, 5))
img = plot_stokes_component(ax[0], channels['S1'])
plt.colorbar(img, ax=ax[0])
img = plot_stokes_component(ax[1], channels['S2'])
plt.colorbar(img, ax=ax[1])
img = plot_stokes_component(ax[2], channels['S3'])
plt.colorbar(img, ax=ax[2])

ax[0].set_xlabel("S1: Horizontal vs. vertical", size=14, weight='bold')
ax[1].set_xlabel("S2: Diagonal", size=14, weight='bold')
ax[2].set_xlabel("S3: Circular", size=14, weight='bold')
plt.savefig('./images/results7.png')