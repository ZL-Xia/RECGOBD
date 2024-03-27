# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pandas as pd

from tensorboard.plugins.hparams import api as hp

from model.metric import calculate_auroc, calculate_aupr,calculate_f1_score,calculate_fmax_score
#from utils.utils import plot_loss_curve, plot_roc_curve, plot_pr_curve
from utils.utils import write2txt, write2csv


physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)



class EarlyStopTrainer(object):
    def __init__(self, model, loss_object, optimizer, hparams,
                 result_dir, checkpoint_dir, summary_dir,
                 epochs=100, patience=5, verbosity=0, tensorboard=True, max_to_keep=5):
        self.model = model
        self.loss_object = loss_object
        self.optimizer = optimizer
        self.hparams = hparams

        self.result_dir = result_dir
        self.summary_dir = summary_dir
        self.checkpoint_dir = checkpoint_dir

        self.epochs = epochs
        self.patience = patience
        self.verbosity = verbosity
        self.tensorboard = tensorboard
        self.max_to_keep = max_to_keep

        if self.verbosity == 0:
            self.dis_show_bar = True
        else:
            self.dis_show_bar = False

        # Initialize the Metrics.
        self.metric_tra_loss = tf.keras.metrics.Mean()
        self.metric_val_loss = tf.keras.metrics.Mean()

        # Initialize the SummaryWriter.
        self.writer = tf.summary.create_file_writer(
            logdir=self.summary_dir)

        # Initialize the CheckpointManager
        self.ckpt = tf.train.Checkpoint(
            step=tf.Variable(0, dtype=tf.int64),
            net=self.model,
            optimizer=self.optimizer)
        self.manager = tf.train.CheckpointManager(
            checkpoint=self.ckpt,
            directory=self.checkpoint_dir,
            max_to_keep=self.max_to_keep)



    def train_step(self, inputs, y):
        with tf.GradientTape() as tape:

            # Forward pass through the model
            predictions = self.model(inputs=inputs, training=True)

            print('y shape:', y.shape)
            print('predictions shape:', predictions.shape)
            
            # Compute the loss
            loss = self.loss_object(y_true=y, y_pred=predictions)
            loss = loss + tf.reduce_sum(self.model.losses)
        
        # Compute gradients and update model parameters
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss, predictions

    
    def train(self, dataset_train, dataset_valid, train_steps, valid_steps):
        print('Begin to train the model.', flush=True)
        
        best_avg_auroc_list = 0
        best_valid_loss = np.inf
        patience_temp = 0
        history = {'epoch': [], 'train_loss': [], 'valid_loss': []}

        for epoch in range(1, self.epochs+1):
            start_time = time.time()
            
            # Initialize the iterator for training dataset
            train_iterator = iter(dataset_train)
            
            with tqdm(range(train_steps), ascii=True, disable=self.dis_show_bar) as pbar:
                for _ in pbar:
                    try:
                        batch_x, batch_y = next(train_iterator)
                    except StopIteration:
                        train_iterator = iter(dataset_train)
                        batch_x, batch_y = next(train_iterator)
                    
                    train_loss, predictions = self.train_step(batch_x, batch_y)
                    batch_size = tf.shape(batch_x[0])[0]
                    self.metric_tra_loss.update_state(train_loss, batch_size)
                    pbar.set_description('Train loss: {:.4f}'.format(train_loss))
                    
            results = []
            labels = []
            with tqdm(range(valid_steps), ascii=True, disable=self.dis_show_bar) as pbar:
                for _, (batch_x, batch_y) in zip(pbar, dataset_valid):
                    predictions = self.model(inputs=batch_x, training=False)
                    valid_loss = self.loss_object(y_true=batch_y, y_pred=predictions)
                    batch_size = tf.shape(batch_x[0])[0]
                    self.metric_val_loss.update_state(valid_loss, batch_size)            
                    pbar.set_description('Valid loss: {:.4f}'.format(valid_loss))       
                    
                     # Calculate AUROC and AUPR for each label
                    results.append(predictions)
                    labels.append(batch_y)
            end_time = time.time()
                
            result = np.concatenate(results)
            label = np.concatenate(labels)
            # Evaluate the result.
            result_shape = np.shape(result)
            # print(result_shape)
            fpr_list, tpr_list, auroc_list = [], [], []
            precision_list, recall_list, aupr_list = [], [], []
            f1_precision_list,f1_recall_list, f1_scores_list = [], [], []
            fm_precision_list,fm_recall_list, fmax_scores_list = [], [], []
            
            for i in range(result_shape[1]):
                fpr_temp, tpr_temp, auroc_temp = calculate_auroc(result[:, i], label[:, i])
                precision_temp, recall_temp, aupr_temp = calculate_aupr(result[:, i], label[:, i])

                fpr_list.append(np.mean(fpr_temp))
                tpr_list.append(np.mean(tpr_temp))
                precision_list.append(precision_temp)
                recall_list.append(recall_temp)
                auroc_list.append(auroc_temp)
                aupr_list.append(aupr_temp)
                
                            
                f1_precision,f1_recall,f1_score = calculate_f1_score(result[:, i], label[:, i])
                fm_precision,fm_recall,fmax_score = calculate_fmax_score(result[:, i], label[:, i], beta=0.5)
                f1_precision_list.append(f1_precision)
                f1_recall_list.append(f1_recall)
                f1_scores_list.append(f1_score)
                fm_precision_list.append(fm_precision)
                fm_recall_list.append(fm_recall)
                fmax_scores_list.append(fmax_score)

                # Print the AUROC and AUPR for each label.
                print('Label {}: AUROC = {:.5f}, AUPR = {:.5f},  F1_score = {:.5f},  Fmax_score = {:.5f}'.format(i, auroc_temp, aupr_temp,f1_score,fmax_score))



            # Write the results to file.
            header = np.array([['label', 'auroc', 'aupr','f1','fmax']])
            content = np.stack((np.arange(result_shape[1]), auroc_list, aupr_list,f1_scores_list,fmax_scores_list), axis=1)
            content = np.concatenate((header, content), axis=0)
        

            # Calculate the average AUROC and AUPR for each label.
            avg_auroc_list = np.nanmean(auroc_list, axis=0)
            avg_aupr_list = np.nanmean(aupr_list, axis=0)
            avg_f1_score = np.nanmean(f1_scores_list,axis=0)
            avg_fmax_score = np.nanmean(fmax_scores_list,axis=0)

            message = 'valid_AVG-AUROC:{:.5f}, valid_AVG-AUPR:{:.5f}.\n'.format(avg_auroc_list, avg_aupr_list)
            message += 'valid_AVG-F1_score:{:.5f}, valid_AVG-Fmax_score:{:.5f}.\n'.format(avg_f1_score, avg_fmax_score)
            print(message)
            
            
            

            epoch_time = end_time - start_time
            real_epoch = self.ckpt.step.assign_add(1)
            epoch_train_loss = self.metric_tra_loss.result()
            epoch_valid_loss = self.metric_val_loss.result()
            history['epoch'].append(real_epoch.numpy())
            history['train_loss'].append(epoch_train_loss.numpy())
            history['valid_loss'].append(epoch_valid_loss.numpy())
            print("Epoch: {} | Train Loss: {:.5f}".format(real_epoch.numpy(), epoch_train_loss.numpy()), flush=True)
            print("Epoch: {} | Valid Loss: {:.5f}".format(real_epoch.numpy(), epoch_valid_loss.numpy()), flush=True)
            print("Epoch: {} | Cost time: {:.5f}: second".format(real_epoch.numpy(), epoch_time), flush=True)
            self.metric_tra_loss.reset_states()
            self.metric_val_loss.reset_states()


            # Save the checkpoint. (Only save the best performance checkpoints)
            if avg_auroc_list > best_avg_auroc_list:
                best_avg_auroc_list = avg_auroc_list
                patience_temp = 0
                save_path = self.manager.save(checkpoint_number=real_epoch)
                write2csv(content, os.path.join(self.result_dir, 'valid_result.csv'))

                write2txt([message], os.path.join(self.result_dir, 'valid_result.txt'))
                print("Saved checkpoint for epoch {}: {}".format(real_epoch.numpy(), save_path), flush=True)
            else:
                patience_temp += 1

            # Early Stop the training loop, if the validation loss didn't decrease for patience epochs.
            if patience_temp == self.patience:
                print('Validation dice has not improved in {} epochs. Stopped training.'
                    .format(self.patience), flush=True)
                break
            
            
        # Plot the loss curve of training and validation, and save the loss value of training and validation.
        print('History dict: ', history, flush=True)
        np.save(os.path.join(self.result_dir, 'history.npy'), history)

        return history

    def test(self, dataset_test, test_steps):
        if self.manager.latest_checkpoint:
            self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
            print("Restored from {}".format(self.manager.latest_checkpoint), flush=True)
        else:
            print("Initializing from scratch.", flush=True)

        results = []
        labels = []
        with tqdm(range(test_steps), ascii=True, disable=self.dis_show_bar, desc='Testing ... ') as pbar:
            for i, (batch_x, batch_y) in zip(pbar, dataset_test):
                predictions = self.model(batch_x, training=False)
                results.append(predictions)
                labels.append(batch_y)

        result = np.concatenate(results)
        label = np.concatenate(labels)

        # Ensure we only process the first ten labels if there are more than ten
        num_labels = min(result.shape[1], 10)
        result = result[:, :num_labels]
        label = label[:, :num_labels]

        # Create a DataFrame
        pred_columns = [f'Pred_Label_{i}' for i in range(num_labels)]
        label_columns = [f'True_Label_{i}' for i in range(num_labels)]
        df_preds = pd.DataFrame(result, columns=pred_columns)
        df_labels = pd.DataFrame(label, columns=label_columns)
        df = pd.concat([df_preds, df_labels], axis=1)

        # Save to CSV
        df.to_csv('bert_pre_and_label.csv', index=False)
        print("Saved predictions and labels to 'predictions_and_labels.csv'")

        # Evaluate the result.
        result_shape = np.shape(result)
        # print(result_shape)
        fpr_list, tpr_list, auroc_list = [], [], []
        precision_list, recall_list, aupr_list = [], [], []
        f1_precision_list,f1_recall_list, f1_scores_list = [], [], []
        fm_precision_list,fm_recall_list, fmax_scores_list = [], [], []
        for i in range(result_shape[1]):
            fpr_temp, tpr_temp, auroc_temp = calculate_auroc(result[:, i], label[:, i])
            precision_temp, recall_temp, aupr_temp = calculate_aupr(result[:, i], label[:, i])

            fpr_list.append(fpr_temp)
            tpr_list.append(tpr_temp)
            precision_list.append(precision_temp)
            recall_list.append(recall_temp)
            auroc_list.append(auroc_temp)
            aupr_list.append(aupr_temp)
            
                        
            f1_precision,f1_recall,f1_score = calculate_f1_score(result[:, i], label[:, i])
            fm_precision,fm_recall,fmax_score = calculate_fmax_score(result[:, i], label[:, i], beta=0.5)
            f1_precision_list.append(f1_precision)
            f1_recall_list.append(f1_recall)
            f1_scores_list.append(f1_score)
            fm_precision_list.append(fm_precision)
            fm_recall_list.append(fm_recall)
            fmax_scores_list.append(fmax_score)

            # Print the AUROC and AUPR for each label.
            print('Label {}: AUROC = {:.5f}, AUPR = {:.5f},  F1_score = {:.5f},  Fmax_score = {:.5f}'.format(i, auroc_temp, aupr_temp,f1_score,fmax_score))

        # Write the results to file.
        header = np.array([['label', 'auroc', 'aupr','f1','fmax']])
        content = np.stack((np.arange(result_shape[1]), auroc_list, aupr_list,f1_scores_list,fmax_scores_list), axis=1)
        content = np.concatenate((header, content), axis=0)
        write2csv(content, os.path.join(self.result_dir, 'result.csv'))

        # Calculate the average AUROC and AUPR for each label.
        avg_auroc_list = np.nanmean(auroc_list, axis=0)
        avg_aupr_list = np.nanmean(aupr_list, axis=0)
        avg_f1_score = np.nanmean(f1_scores_list,axis=0)
        avg_fmax_score = np.nanmean(fmax_scores_list,axis=0)

        message = 'valid_AVG-AUROC:{:.5f}, valid_AVG-AUPR:{:.5f}.\n'.format(avg_auroc_list, avg_aupr_list)
        message += 'valid_AVG-F1_score:{:.5f}, valid_AVG-Fmax_score:{:.5f}.\n'.format(avg_f1_score, avg_fmax_score)
        # message = 'AVG-AUROC:{:.5f}, AVG-AUPR:{:.5f}.'.format(avg_auroc_list, avg_aupr_list)
        write2txt([message], os.path.join(self.result_dir, 'result.txt'))
        print(message)

        return result, label