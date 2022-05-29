import imageio
import align.detect_face

img = imageio.imread('..\data\images\\train\\ben_afflek\httpssmediacacheakpinimgcomxdbbdbbbececacdecdcdfjpg.jpg')

img = img[:,:,0:3]

minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor


bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
nrof_faces = bounding_boxes.shape[0]

print(nrof_faces)
