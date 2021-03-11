# Lane Keeping (Motorway)

So according to [Consumer Reports](https://www.consumerreports.org/car-safety/cadillac-super-cruise-outperforms-other-active-driving-assistance-systems/) Cadillac's Super Cruise is the best driving assistance system so its functionality is the goal.

Using an open source data-set lets make a perception model as a base for are lane keeping assistance, we will use the [Comma10k](https://github.com/commaai/comma10k.git) which is a segmentation data-set with 5 classes.

## Train_segnet Dir

### Data Prep

1) Clone [comma10k](https://github.com/commaai/comma10k.git)
2) Run make_comma10k.py - Converts RGB masks to categorical masks

### Model Training
1) Update config.py to meet the needs for your GPU/CPU
3) Make model dir
2) Run train.py to train the model

