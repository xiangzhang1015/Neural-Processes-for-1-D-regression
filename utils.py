import numpy as np
import tensorflow as tf
np.random.seed(2018)
# define a NP
class NP:
    def __init__(self, r_dim=64,z_dim=16,hidden1=32, hidden2=32, g_hidden=32):
        self.r_dim=r_dim
        self.z_dim=z_dim
        self.hidden1=hidden1
        self.hidden2=hidden2
        self.g_hidden=g_hidden

    def build_model(self,X_content,Y_content,X_target,Y_target,learning_rate=0.001,istrain=True):
        if istrain:
            X_all=tf.concat([X_content,X_target],axis=0)
            Y_all=tf.concat([Y_content,Y_target],axis=0)
            z_con_mu,z_con_sigma=self.h_encoder(X_content,Y_content)
            z_all_mu,z_all_sigma=self.h_encoder(X_all,Y_all)

            self.epsilon=tf.random_normal(shape=(1,16))
            z_sample=tf.add(tf.multiply(self.epsilon, z_all_sigma), z_all_mu)
            mu_star, sigma_star=self.g_decoder(z_sample,X_target)

            #define loss function
            logpro=self.loglikelihood(Y_target,mu_star,sigma_star)
            KL_loss=self.KLD(z_all_mu,z_all_sigma,z_con_mu,z_con_sigma)
            loss=tf.negative(logpro)+KL_loss
            #optimization
            optimizer=tf.train.AdamOptimizer(learning_rate)
            train_op=optimizer.minimize(loss)
            return train_op, loss

    def h_encoder(self,X_input,Y_input):
        New_XY=tf.concat([X_input,Y_input],axis=1)
        layer1=tf.layers.dense(New_XY,self.hidden1,activation=tf.nn.relu,name='encoder_layer1',reuse=tf.AUTO_REUSE)
        layer2=tf.layers.dense(layer1,self.hidden1,activation=tf.nn.relu,name='encoder_layer2',reuse=tf.AUTO_REUSE)
        output1=tf.layers.dense(layer2,self.r_dim,name='output',reuse=tf.AUTO_REUSE)
        aggre1=tf.reduce_mean(output1,axis=0)
        output2=tf.reshape(aggre1,shape=[1,-1])
        #get mu and sigma
        mu=tf.layers.dense(output2, self.z_dim,name='z_para_mu',reuse=tf.AUTO_REUSE)
        sigma=tf.nn.softplus(tf.layers.dense(output2, self.z_dim,name='z_para_sigma',reuse=tf.AUTO_REUSE))
        return mu, sigma

    def g_decoder(self,z_sample,X_inputs,noise=0.05):
        n_draws=np.array(z_sample.get_shape())[0]
        N_star=tf.shape(X_inputs)[0]
        z_sample_rep=tf.expand_dims(z_sample,axis=1)
        z_sample_rep=tf.tile(z_sample_rep,[1,N_star,1])
        X_inputs_rep=tf.expand_dims(X_inputs,axis=0)
        X_inputs_rep=tf.tile(X_inputs_rep,[n_draws,1,1])
        Con_inputs=tf.concat([X_inputs_rep,z_sample_rep],axis=2)
        output1=tf.layers.dense(Con_inputs,self.g_hidden, tf.nn.sigmoid,name='decoder_layer1',reuse=tf.AUTO_REUSE)
        mu_star=tf.layers.dense(output1,1,name='decoder_layer2',reuse=tf.AUTO_REUSE)
        mu_star=tf.transpose(tf.squeeze(mu_star,axis=2))
        sigma_star=tf.constant(noise,dtype=tf.float32)
        return  mu_star,sigma_star

    def loglikelihood(self, Y_target, mu_star,sigma_star):
        p_normal=tf.distributions.Normal(loc=mu_star,scale=sigma_star)
        loglik=tf.reduce_mean(tf.reduce_sum(p_normal.log_prob(Y_target),axis=0))
        return loglik

    def KLD(self,mu_q,sigma_q,mu_p,sigma_p):
        sigma2_q=tf.add(tf.square(sigma_q),0.00001)
        sigma2_p=tf.add(tf.square(sigma_p),0.00001)
        temp=sigma2_q/sigma2_p+tf.square(mu_q-mu_p)/sigma2_p-1+tf.log(sigma2_p/sigma2_q+0.00001)
        KLD=0.5*tf.reduce_sum(temp)
        return KLD

    def posterior_pred(self,X_content,Y_content,X_target):
        z_params_mu, z_params_sigma = self.h_encoder(X_content, Y_content)
        z_sample = tf.add(tf.multiply(self.epsilon, z_params_sigma), z_params_mu)
        y_pred_tgt1, y_pred_tgt2 = self.g_decoder(z_sample, X_target)
        return y_pred_tgt1

    # def posterior_pred(self,x,y,x_test):
    #     x_obs=tf.constant(x,dtype=tf.float32)
    #     y_obs=tf.constant(y, dtype=tf.float32)
    #     x_test_star = tf.constant(x_test, dtype=tf.float32)
    #     z_params_mu,z_params_sigma=self.h_encoder(x_obs,y_obs)
    #     z_sample = tf.add(tf.multiply(self.epsilon, z_params_sigma), z_params_mu)
    #     y_pred_tgt1,y_pred_tgt2=self.g_decoder(z_sample,x_test_star)
    #     return y_pred_tgt1


def train_ct(X_data,Y_data):
    N=np.random.randint(10,50,1)[0]
    #N=30
    X_c=X_data[:N][:,np.newaxis]
    Y_c=Y_data[:N][:,np.newaxis]
    X_t=X_data[N:][:,np.newaxis]
    Y_t=Y_data[N:][:,np.newaxis]
    return X_c,Y_c,X_t,Y_t
