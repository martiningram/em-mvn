# EM for mixture of multivariate normals

A simple implementation of expectation maximisation for a mixture of multivariate normal distributions. For details, see e.g. Chapter 9 in [Christopher Bishop's Pattern Recognition and Machine Learning book](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf).

There are just two files in this repository:

* `mvn_mix_em.py` contains the actual EM implementation for a mixture of
  multivariate normals. The key function is `fit_em`. I only tried this on toy examples, but I expect it should work just fine on real problems.
* `2D Illustration.ipynb` is a jupyter notebook that runs EM on a toy example in
  2D. It also has some code to generate a sequence of images to illustrate
  convergence. The PNGs can be combined e.g. by using ImageMagick:

  `convert individual_images/*.png animation.gif`

  to produce a GIF like the one below.

![animation](animation.gif)

