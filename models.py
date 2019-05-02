import tensorflow as tf
import numpy as np
import os
def pe_encoding(mixture,phase):
    return input_feat

def multi_head(Q, K, V,basic):
    return input_feat

def feed_forward(input_feat):
    return input_feat

class transformer(object):
    def __init__(self, config):
        self.makeplaceholder()
        self.optimizer()
        #self.saver = 
    def encoder(self, input_feat):
        input_feat = pe_encoding(input_feat)
        for i in range(self.num):
            input_feat = multi_head(Q,K,V, input_feat)
            input_feat = feed_forward(input_feat)
        return input_feat

    def decoder(self, input_feat, memory):
        input_feat = pe_encoding(input_feat)
        for i in range(self.num):
            input_feat = multi_head(memory, memory, input_feat, input_feat)
            input_feat = feed_forward(input_feat)
        return input_feat

    def tower_cost(self,input_feat, output_feat):
        mem = self.encoder(input_feat)
        out = self.decoder(input_feat, mem)
        

    def optimizer(self):
        optimizer = tf.train.adamoptimizer(learning_rate = self.learning_rate)
        loss = self.tower_cost(self.input_feat, self.output_feat)
        grads = optimizer.compute_gradients(loss)
        self.train_op = optimizer.apply_gradients(grads)


