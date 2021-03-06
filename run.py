import src.main as main

if __name__ == '__main__':
    model = main.Main(practie_name="xception-finetuning-t-test",
                      model_name="xception",
                      img_height=235,
                      img_width=80,
                      t_learning=True,
                      fine_tuning=False)

    model.images_create()
    model.train(batch_size=128, epoch_size=300)
    # model.train_nfold(n_splits=5, batch_size=128, epoch_size=100)
    # weights = ["xception-80-128-100-hsv"]
    # model.test(weights)