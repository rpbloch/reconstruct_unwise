mask = np.zeros(12*64**2)
for ipix in np.arange(mask.size):
    if np.pi/8 <= hp.pix2ang(64,ipix)[0] <= np.pi/3:
            if np.pi/8 <= hp.pix2ang(64,ipix)[1] <= np.pi/2:
                    mask[ipix] = 1

plt.figure()
plt.hist((N1024_masked*mask)[np.where(mask!=0)], bins=100, label='good subsky')

mask = np.zeros(12*64**2)
for ipix in np.arange(mask.size):
    if 12*np.pi/20 <= hp.pix2ang(64,ipix)[0] <= 9*np.pi/12:
            if 0 <= hp.pix2ang(64,ipix)[1] <= np.pi/4:
                    mask[ipix] = 1
            elif 14*np.pi/8 <= hp.pix2ang(64,ipix)[1] <= 2*np.pi:
                    mask[ipix] = 1

plt.hist((N1024_masked*mask)[np.where(mask!=0)], bins=100, label='bad subsky',alpha=0.75)
plt.legend()

plt.savefig('test')
