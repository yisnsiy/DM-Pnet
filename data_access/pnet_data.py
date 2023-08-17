from data_access.data_access import Data


class PnetData:
    def __init__(self, data_params):
        if type(data_params) == list:
            self.data_params = data_params[0]
        else:
            self.data_params = data_params
        print("data_params: ", self.data_params)
        data = Data(**data_params)
        x_train, x_validate_, x_test_, y_train, y_validate_, y_test_, info_train, info_validate_, info_test_, cols = data.get_train_validate_test()

        self.x_train = x_train
        self.x_validate_ = x_validate_
        self.x_test_ = x_test_
        self.y_train = y_train
        self.y_validate_ = y_validate_
        self.y_test_ = y_test_
        self.info_train = info_train
        self.info_validate_ = info_validate_
        self.info_test_ = info_test_
        self.cols = cols
