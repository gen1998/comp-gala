import src.train as train

if __name__ == '__main__':
    model = train.Model(practie_name="vgg16-224-32-100", operation="ubuntu", image_size=224)
    model.images_create()
    model.train(batch_size=32)
    model.test()