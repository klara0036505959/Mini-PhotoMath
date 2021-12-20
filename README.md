<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- ABOUT THE PROJECT -->
## About The Project
<b>Mini PhotoMath Project </b>
* evaluation of handwritten math expressions
* supports all digits ( 0 1 2 3 4 5 6 7 8 9 ) and basic math operators (addition, subtraction, multiplication, division, brackets)
* upload an image to server (https://mini-photomath-klara-marijan.herokuapp.com/) via android app in repo or some other way :)
* server sends back its response (prediction and evaluation of the expression - if it is possible to calculate it)
</p>

<img src="https://github.com/klara0036505959/Mini-PhotoMath/blob/master/proof_of_work_gifs/pm_proof.gif" width="300"/> <img src="https://github.com/klara0036505959/Mini-PhotoMath/blob/master/proof_of_work_gifs/pm_proof_2.gif" width="300"/> 

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

<b>If you want to run this project on your smartphone:</b>
* Android Studio
* JDK 11 (or later)
* Android SDK
note: this can only be possible while heroku server is running


<b>You can also run this project locally on your PC:</b>
* Python >= 3.8
* numpy >= 1.19.5
* tensorflow >= 2.7.0
* opencv-python >= 4.2.0.32
* Keras
* Pillow
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

<b>To test the model on a certain image on your PC:</b>

   ```sh
   python test_model.py [--model_path [MODEL_PATH]] [--img_path [IMAGE_PATH]]
   
    --model_path        Path to the saved model 
                        (some models are already trained, check out trained_models directory in this repo,
                        currently the best performing model is "my_model_good_final.h5")
    --img_path          Path to the image you want to run prediction on 
                        (there are already some test images in the test_images directory in this repo)                             
   ``` 
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Some more notes about the structure of the repository -->
## Some more notes about the structure of the repository 

* <b>jupyter_notebooks: </b> results of testing some images are already visible
* <b>photomath_app_flask: </b> every piece of code needed for putting the model on the cloud (for example: heroku), written using Flask
* <b>version_custom_dataset: </b> model trained on a custom-made dataset (by me)
* <b>version_mnist_kaggle: </b> model trained on a dataset which is made by merging MNIST and a part of a Kaggle dataset (https://www.kaggle.com/xainano/handwrittenmathsymbols)
 
                          

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- Comments & possible improvements -->
## Comments & possible improvements

* find better hyperparameters for this CNN model (random search, grid search or a similar approach)
* make a bigger custom dataset
* model trained on MNIST + Kaggle (60k+ images) is not performing well - find a better model architecture
* metrics used to evaluate model on validation dataset: accuracy -> would be useful to also track recall, precision, F1 score and confusion matrix
                          

<p align="right">(<a href="#top">back to top</a>)</p>




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
