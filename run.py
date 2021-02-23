import src.main as main

if __name__ == '__main__':
    model = main.Main(practie_name="vgg16-80-128-100-hsv",
                      model_name="vgg16",
                      image_size_x=80,
                      image_size_y=80)
    model.images_create()
    model.train(batch_size=128, epoch_size=100)
    model.test()