# quick-draw-backend

## Objective
The trained neural network model should predict the drawn image correctly

## Clone this repository

```
git clone https://github.com/Adityaupadhye/quick-draw-recogniser.git
cd quick-draw-recogniser
```

## Pip Installation
```
pip install -r requirements.txt
```

## Results
<img src="/results/resultss.png" width="60%" height="60%">

## How to train the model

```
python model_train.py
```

## Dataset structure
```
data
├── Bird
│   └── image
│       ├── Bird_1613020596973.png
│       ├── Bird_1613020665245.png
│       ├── Bird_1613021591310.png
│       ├── Bird_1613032256287.png
│       ├── Bird_1613032320069.png
│       ├── Bird_1613048236635.png
│       ├── Bird_1613048290136.png
│       ├── Bird_1613048336656.png
│       ├── Bird_1613048430073.png
│       └── Bird_1613048472214.png
├── Flower
│   └── image
│       ├── Flower_1613021163209.png
│       ├── Flower_1613021220324.png
│       ├── Flower_1613021251673.png
│       ├── Flower_1613021288600.png

│       ├── Flower_1613021533630.png
│       ├── Flower_1613031084396.png
│       ├── Flower_1613031568962.png
│       ├── Flower_1613037250037.png
│       ├── Flower_1613037316738.png
│       └── Flower_1613046529258.png
├── Hand
│   └── image
│       ├── Hand_1612968687087.png
│       ├── Hand_1612968751876.png
│       ├── Hand_1613020741714.png
│       ├── Hand_1613046686509.png
│       ├── Hand_1613046709464.png
│       ├── Hand_1613046727731.png
│       ├── Hand_1613046744480.png
│       ├── Hand_1613046808161.png
│       ├── Hand_1613046878740.png
│       └── Hand_1613046913427.png
├── House
│   └── image
│       ├── House_1612958523366.png
│       ├── House_1613020818678.png
│       ├── House_1613032374534.png
│       ├── House_1613032409763.png
│       ├── House_1613035542064.png
│       ├── House_1613047011146.png
│       ├── House_1613047141844.png
│       ├── House_1613047201571.png
│       ├── House_1613047285747.png
│       └── House_1613047315313.png
├── Pencil
│   └── image
│       ├── Pencil_1613021118160.png
│       ├── Pencil_1613031682707.png
│       ├── Pencil_1613037441573.png
│       ├── Pencil_1613047370472.png
│       ├── Pencil_1613047397458.png
│       ├── Pencil_1613047422182.png
│       ├── Pencil_1613047446316.png
│       ├── Pencil_1613047474463.png
│       ├── Pencil_1613047512282.png
│       └── Pencil_1613047550421.png
├── Spectacles
│   └── image
│       ├── Spectacles_1612958478329.png
│       ├── Spectacles_1612969027031.png
│       ├── Spectacles_1612969083114.png
│       ├── Spectacles_1613020852766.png
│       ├── Spectacles_1613020895908.png
│       ├── Spectacles_1613031285886.png
│       ├── Spectacles_1613032162424.png
│       ├── Spectacles_1613047671022.png
│       ├── Spectacles_1613047702842.png
│       └── Spectacles_1613047740653.png
├── Spoon
│   └── image
│       ├── Spoon_1612968878800.png
│       ├── Spoon_1612968894143.png
│       ├── Spoon_1612968952536.png
│       ├── Spoon_1612968985835.png
│       ├── Spoon_1613021012896.png
│       ├── Spoon_1613031962703.png
│       ├── Spoon_1613047782332.png
│       ├── Spoon_1613047799857.png
│       ├── Spoon_1613047820833.png
│       └── Spoon_1613047848629.png
├── Sun
│   └── image
│       ├── Sun_1612958252771.png
│       ├── Sun_1612969138310.png
│       ├── Sun_1613019922698.png
│       ├── Sun_1613020517499.png
│       ├── Sun_1613020563963.png
│       ├── Sun_1613031186103.png
│       ├── Sun_1613031512767.png
│       ├── Sun_1613047892327.png
│       ├── Sun_1613047922545.png
│       └── Sun_1613047960031.png
├── Tree
│   └── image
│       ├── Tree_1612958424030.png
│       ├── Tree_1612970057695.png
│       ├── Tree_1612970143668.png
│       ├── Tree_1613020919259.png
│       ├── Tree_1613020934522.png
│       ├── Tree_1613020956732.png
│       ├── Tree_1613020979652.png
│       ├── Tree_1613023995310.png
│       ├── Tree_1613032026593.png
│       ├── Tree_1613032070334.png
│       ├── Tree_1613035664122.png
│       └── Tree_1613035767678.png
└── Umbrella
    └── image
        ├── Umbrella_1612969373380.png
        ├── Umbrella_1612969775350.png
        ├── Umbrella_1613021311769.png
        ├── Umbrella_1613021329357.png
        ├── Umbrella_1613031662922.png
        ├── Umbrella_1613037568548.png
        ├── Umbrella_1613048017183.png
        ├── Umbrella_1613048038088.png
        ├── Umbrella_1613048079244.png
        └── Umbrella_1613048100459.png
```
