COCO2BOREAS = {
    3:  0, # car
    1:  1, # person
    10: 2, # traffic light
    8:  3, # truck
    2:  4, # bicycle
}

# map 5 labels of Boreas to 1K labels of ImageNet dataset
# reference: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
BOREAS2IMAGENET = {
    # car
    0: [
        705, # passenger car, coach, carriage
        717, # pickup, pickup truck
        751, # racer, race car, racing car
        817, # sports car, sport car
        829, # streetcar, tram, tramcar, trolley, trolley car
        864, # tow truck, tow car, wrecker
    ],

    # person      
    1: None,

    # traffic light
    2: [
        920, # traffic light, traffic signal, stoplight
    ],

    # truck
    3: [
        555, # fire engine, fire truck
        569, # garbage truck, dustcart
        867, # trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi
    ],

    # bicycle
    4: [
        444, # bicycle-built-for-two, tandem bicycle, tandem
    ],
}