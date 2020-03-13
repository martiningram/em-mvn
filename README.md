# EM for mixture of multivariate normals

Just a quick implementation of EM.

There are just two files in this repository:

* `mvn_mix_em.py` contains the actual EM implementation for a mixture of
  multivariate normals. The key function is `fit_em`. I only tried this on toy examples, but I expect it should work just fine on real problems.
* `2D Illustration.ipynb` is a jupyter notebook that runs EM on a toy example in
  2D. It also has some code to generate a sequence of images to illustrate
  convergence. The PNGs can be combined e.g. by using ImageMagick:

  `convert individual_images/*.png animation.gif`

  to produce a GIF like the one below.

![animation](animation.gif)

