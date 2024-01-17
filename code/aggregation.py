import torch
import tenseal as ts
import time


# HE_FL
class ModelEncryptor():
    
    def __init__(self, N, poly_mod_degree=4096, coeff_mod_bit_sizes=[18,18,18]):
        start = time.time()
        # 初始化并生成密钥
        self.ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
        self.ctx.global_scale = 2 ** 18
        self.ctx.generate_galois_keys()
        end = time.time()
        print(f"密钥生成时间:", float(end - start))

        self.N = N
        self.shapes = {}
        

    def encrypt_model(self, filename):
        # model = CNN14BUS.VGG16()
        # model.load_state_dict(torch.load(filename))
        # model_dict = model.state_dict()
        model_dict = torch.load(filename,map_location=torch.device('cpu'))

        shape = {}
        for key, value in model_dict.items():
            # 存储当前参数的形状
            shape[key] = value.shape
        self.shapes = shape

        start = time.time()

        encrypted_params = {}
        encrypted_params['feature.0.weight'] = ts.ckks_tensor(self.ctx, model_dict['feature.0.weight'])

        encrypted_params['feature.0.bias'] = ts.ckks_tensor(self.ctx, model_dict['feature.0.bias'])

        encrypted_params['feature.4.weight'] = ts.ckks_tensor(self.ctx, model_dict['feature.4.weight'])

        encrypted_params['feature.4.bias'] = ts.ckks_tensor(self.ctx, model_dict['feature.4.bias'])

        encrypted_params['feature.8.weight'] = ts.ckks_tensor(self.ctx, model_dict['feature.8.weight'])

        encrypted_params['feature.8.bias'] = ts.ckks_tensor(self.ctx, model_dict['feature.8.bias'])

        encrypted_params['classifer.0.weight'] = ts.ckks_tensor(self.ctx, model_dict['classifer.0.weight'])

        encrypted_params['classifer.0.bias'] = ts.ckks_tensor(self.ctx, model_dict['classifer.0.bias'])

        encrypted_params['classifer.3.weight'] = ts.ckks_tensor(self.ctx, model_dict['classifer.3.weight'])
        encrypted_params['classifer.3.bias'] = ts.ckks_tensor(self.ctx, model_dict['classifer.3.bias'])
        
        end = time.time()
        print(f"加密路径{filename} 时间:", float(end - start))

        return encrypted_params


    def aggregate_encrypted_models(self, model_paths):
        encrypted_models = {}

        # 对于每个模型路径，加密该模型
        for path in model_paths:
            encrypted_models[path] = self.encrypt_model(path)

        # 聚合所有加密模型的参数
        aggregated_params = {key: encrypted_models[model_paths[0]][key] for key in encrypted_models[model_paths[0]]}
        for path in model_paths[1:]:
            for key in aggregated_params:
                aggregated_params[key] += encrypted_models[path][key]

        # # 对聚合的参数进行平均
        # for key in aggregated_params:
        #     aggregated_params[key] = aggregated_params[key] * (1/N)

        return aggregated_params


    def decrypt_params(self, aggregated_params):
        decrypted_params = {}
        # 解密参数并保存到指定的路径
        for key, encrypted_tensor in aggregated_params.items():
            decrypted_params[key] = torch.Tensor(encrypted_tensor.decrypt().raw)

        # 对聚合的参数进行平均
        for key in decrypted_params:
            decrypted_params[key] = decrypted_params[key] / self.N

        return decrypted_params


    def save_decrypted_model(self, aggregated_params, save_path):
        # 解密参数并保存到指定的路径
        decrypted_params_reshape = {}
        decrypted_params = self.decrypt_params(aggregated_params)
        for key, decrypted_param in decrypted_params.items():
            decrypted_params_reshape[key] = decrypted_param.reshape(self.shapes[key])
        torch.save(decrypted_params_reshape, save_path)

# FL
class ModelAggre():
    def __init__(self, N):
        self.N = N

    
    def get_models(self,filename):
        
        model_dict = torch.load(filename,map_location=torch.device('cpu'))

        model_params = {}
        model_params['feature.0.weight'] = model_dict['feature.0.weight']

        model_params['feature.0.bias'] = model_dict['feature.0.bias']

        model_params['feature.4.weight'] = model_dict['feature.4.weight']

        model_params['feature.4.bias'] = model_dict['feature.4.bias']

        model_params['feature.8.weight'] = model_dict['feature.8.weight']

        model_params['feature.8.bias'] = model_dict['feature.8.bias']

        model_params['classifer.0.weight'] = model_dict['classifer.0.weight']
        
        model_params['classifer.0.bias'] = model_dict['classifer.0.bias']

        model_params['classifer.3.weight'] = model_dict['classifer.3.weight']
        model_params['classifer.3.bias'] = model_dict['classifer.3.bias']
        

        return model_params
    
    def aggregate_models(self, model_paths):
        
        models = {}
        for path in model_paths:
            models[path] = self.get_models(path)

        aggregated_params = {key: models[model_paths[0]][key] for key in models[model_paths[0]]}

        # For each model, add its parameters to our aggregated_params
        for path in model_paths[1:]:
            for key in aggregated_params:
                aggregated_params[key] += models[path][key]


        # After summing all model parameters, divide by N to get the average
        for key in aggregated_params:
            aggregated_params[key] /= self.N

        return aggregated_params

    def save_aggregated_model(self, model_paths, save_path):
        
        averaged_params = self.aggregate_models(model_paths)
        torch.save(averaged_params, save_path)
        
    
if __name__ == '__main__':
    # 用户数
    N = 2
    model_paths = [f'pth/part{i}_14.pth' for i in range(1, N+1)]
    # 创建加密器对象  
    print('创建加密器对象')
    encryptor = ModelEncryptor(N)
    print('完成')
    

    # 使用加密器聚合和平均模型参数
    print('加密、聚合和平均模型参数')
    averaged_params = encryptor.aggregate_encrypted_models(model_paths)
    print('完成')

    # 指定解密后模型的保存路径，并保存
    print('解密后模型并保存')
    save_path = "pth/averaged_decrypted_model_14.pth"
    encryptor.save_decrypted_model(averaged_params, save_path)
    print('完成')
    

