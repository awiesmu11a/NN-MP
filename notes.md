# Motion Planning via Reinforcement Learning - F1-tenth car

* We will need to generalise the network to be able to test any kind of network
* CNN and FCN files in the network have the dimension of the input hardcoded. This can be changed. Look for generalising this.
* We'll design and test a model in simulation. Next step would be to create a ONNX model and use it for car.
* In the current workflow of the car, we build the ROS workspace directly and all the values are hardcoded, including the size of image and path of the models.
* Make two different test. One for testing the building of model and other about the ROS workspace. 

## TODO:

* Make a framework to test any kind of model without using ROS. Just testing whether model is loaded as engine or not.
* In the above framework include a config file to enter paths, sample input, and required shapes of the input.
* Now can make a final framework for ROS interface which can be deployed directly on the car.
* Here also add a config file for the model path, input shapes, any other resizing, topic and node names. 
* Create a common config file.