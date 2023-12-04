# CS 445 Final Project

In this final project, we aim to incorporate additional techniques not yet covered in class, such as object averaging (morphing), image warping, matting, and hole-filling, to create a pipeline for “Dynamic Object & Scene Migration in Classical Artwork.”

Our inspiration comes from a creative rendition of Vermeer’s “Girl with a Pearl Earring” (shown in Figure 1), where the girl's face is replaced with that of a cartoon cat. This gave us the idea to seamlessly merge objects from one classical artwork into the scene or objects of another. Our primary objective is to facilitate face-to-face swaps that do more than just blend faces; we also aim to apply texture transfer to match the drawing style and morph only into the face region, preserving the integrity of the original artwork. Figure 2 shows another example we aim to achieve, swapping the two faces in “American Gothic” by Grant Wood. Additionally, if time permits, we would like to create an animation demonstrating the transition from the original artwork to the modified image.

![Figure 1](./docs/Screenshot%202023-12-03%20at%2023.32.45.png)

## Pipeline

![Pipeline](./docs/Screenshot%202023-12-03%20at%2023.35.50.png)

## File Structure

```shell
.
├── README.md
├── docs # assets (e.g. images) for README.md
├── images # images used in the project
├── landmarks # landmarks for images, stored in JSON format
│   └── pexels-photo-736532.json # this is an already annotated image for `pexels-photo-736532.jpg`
├── predictor
│   └── shape_predictor_68_face_landmarks.dat # face landmark predictor from `dlib``
├── notebook.ipynb # main notebook for the project
├── manual_cat_extraction.py # manually extract cat face from image
└── utils.py # utility functions imported in notebook.ipynb
```

## Pre-requisites

Install all neccessary packages by running the following command:

```shell
pip install -r requirements.txt
```

## Usage

To run the notebook, launch your Jupyter Notebook and open `notebook.ipynb`.

### Extract Cat Face Landmarks (68)

```shell
# Usage: manual_cat_extraction.py [-h] [--json_path JSON_PATH] image_path
# Example:
python3.10 manual_cat_extraction.py images/pexels-photo-736532.jpeg --json_path landmarks/pexels-photo-736532.json
```

- Use `mouse left click` to select a point
  - the point will be added to the list of landmarks
  - once the list reaches 68 points, the program will save the list to `points.json`
- Use `mouse right click` to move adjust of point to a new location
  - everytime you update the list, the program update the point you are currently adjusting; and update the `points.json`
- Press `q` to quit

## To-do

- [ ] Style transfer
- [ ] Collec more images

## Reference

- [`dlib`'s demo code for using 68 landmarks predictor](http://dlib.net/face_landmark_detection.py.html)
- Relevant OpenCV function docs 
  - [`seamlessClone()`](https://docs.opencv.org/3.4/df/da0/group__photo__clone.html#ga2bf426e4c93a6b1f21705513dfeca49d)
  - [`Subdiv2D()`](https://docs.opencv.org/3.4/df/d5b/group__imgproc__subdiv2d.html)
- CatFLW: Cat Facial Landmarks in the Wild Dataset.
  - It detects up to 48 landmarks on cat faces. But unfortunately, it is not publicly available.
  - Finka, L. R., Luna, S. P., Brondani, J. T., Tzimiropoulos, Y., McDonagh, J., Farnworth, M. J., Ruta, M., and Mills, D. S., "Geometric morphometrics for the study of facial expressions in non-human animals, using the domestic cat as an exemplar," *Scientific Reports*, vol. 9, no. 1, Art. no. 9883, Jul. 2019, doi: [10.1038/s41598-019-46330-5](https://doi.org/10.1038/s41598-019-46330-5).
  - Martvel, G., Shimshoni, I., and Zamansky, A., "Automated Detection of Cat Facial Landmarks," 2023, arXiv: [2310.09793](https://arxiv.org/abs/2310.09793) [cs.CV].
  - Martvel, G., Farhat, N., Shimshoni, I., and Zamansky, A., "CatFLW: Cat Facial Landmarks in the Wild Dataset," 2023, arXiv: [2305.04232](https://arxiv.org/abs/2305.04232) [cs.CV].
  - Tech4Animals Lab, 'CatFLW: Cat Facial Landmarks in the Wild Dataset,' 2023, Tech4Animals, Available: https://www.tech4animals.org/catflw.
- Other relevant projects/articles:
  - [Landmark Detection for Animal Face and 3D Reconstructions](https://zhangtemplar.github.io/animal-keypoints/) by Qiang Zhang
  - [Face Swap using OpenCV](https://learnopencv.com/face-swap-using-opencv-c-python/) by Satya Mallick