import src.train as train

if __name__ == '__main__':
    model = train.Model(practie_name="vgg16-80-128-200", operation="ubuntu", image_size_x=80, image_size_y=80)
    model.images_create()
    model.train(batch_size=128)
    model.test()