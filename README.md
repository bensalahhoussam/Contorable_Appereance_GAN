# Contorable_Appereance_GAN

## Sports Pose Video Generation

This project combines pose prediction for sports activities with style-preserving image generation to create realistic video sequences of human movements.

The system takes a single source image of a person and generates a sequence of frames showing that person performing predicted sports poses, while maintaining the original style and appearance.

### Features

Pose Prediction: Predicts future poses of a person performing a sports activity.

Style-preserving Image Generation: Uses the predicted poses and a source image to generate new frames while maintaining the source person's style and appearance.

Video Generation: Combines generated frames into a smooth video of the predicted motion.

### How It Works


It has two model , one to generated future pose for k frames and second using image source as style info and pose to generate image that content same style and same pose 

Inputs:

* A source image of a person.

* A Target pose 

Pose Prediction:

* A first model predicts the future sequence of poses for the activity.

Image Generation:

Using the predicted poses and the source image, the generator creates new images.

The generator ensures that the generated images keep the style, clothing, and identity of the source.

Video Compilation:

Generated images are combined into a video showing smooth motion.

## results 

For the predicition pose model 

![pose_prediction_1_20](https://github.com/user-attachments/assets/4fc58405-a3e5-4b4e-af2d-c22c3214687c)


![pose_prediction_1_29](https://github.com/user-attachments/assets/0b894bb2-ffc1-4eff-bd1a-80a463bfb59f)

for image generator 

* First 4 images are the sources 
* second 4 images are the generated images 
* third 4 images are the target pose 

![epoch_031_comparison](https://github.com/user-attachments/assets/6ca769cf-d4ae-464e-a395-82a20c354651)



![epoch_030_comparison](https://github.com/user-attachments/assets/3ce1b78a-d6cc-4b77-978d-ef308367f7de)





## for the upcoming results 

Gan will takes time to generate good results , and then will generate videos based on psoe prediction 
