import os

from pathlib import Path

import numpy as np
import tensorflow as tf

import os
import time
from functools import partial

import keras
import yaml
from keras import backend as K
from keras.callbacks import CSVLogger
from keras.layers import Lambda
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop

import models.charvae.src.hyperparameters as hyperparameters
import models.charvae.src.mol_callbacks as mol_cb
import models.charvae.src.mol_utils as mu
from models.charvae.src.models import (
    decoder_model,
    encoder_model,
    load_decoder,
    load_encoder,
    load_property_predictor,
    property_predictor_model,
    variational_layers,
    CHARS
)
from models.global_utils import get_model_config, BASELINE_DIR, WB_LOG_DIR, SMILES_DIR, DATA_DIR


def vectorize_data(params, dataset):
    MAX_LEN = params["MAX_LEN"]
    path_to_char_file = DATA_DIR / "CHARVAE" / dataset / "chars.json"
    params["NCHARS"] = len(CHARS)
    NCHARS = len(CHARS)
    CHAR_INDICES = dict((c, i) for i, c in enumerate(CHARS))

    ## Load data for properties
    if params["do_prop_pred"] and ("data_file" in params):
        if "data_normalization_out" in params:
            normalize_out = params["data_normalization_out"]
        else:
            normalize_out = None

        ################
        if ("reg_prop_tasks" in params) and ("logit_prop_tasks" in params):
            smiles, Y_reg, Y_logit = mu.load_smiles_and_data_df(
                params["data_file"],
                MAX_LEN,
                reg_tasks=params["reg_prop_tasks"],
                logit_tasks=params["logit_prop_tasks"],
                normalize_out=normalize_out,
            )
        elif "logit_prop_tasks" in params:
            smiles, Y_logit = mu.load_smiles_and_data_df(
                params["data_file"],
                MAX_LEN,
                logit_tasks=params["logit_prop_tasks"],
                normalize_out=normalize_out,
            )
        elif "reg_prop_tasks" in params:
            smiles, Y_reg = mu.load_smiles_and_data_df(
                params["data_file"],
                MAX_LEN,
                reg_tasks=params["reg_prop_tasks"],
                normalize_out=normalize_out,
            )
        else:
            raise ValueError("please sepcify logit and/or reg tasks")

    ## Load data if no properties
    else:
        # smiles = mu.load_smiles_and_data_df(params["data_file"], MAX_LEN)
        path_to_train_file = SMILES_DIR / dataset / "train.txt"
        with open(path_to_train_file) as file:
            smiles = file.readlines()
        smiles = [s.strip("\n") for s in smiles]

    if "limit_data" in params.keys():
        sample_idx = np.random.choice(np.arange(len(smiles)), params["limit_data"], replace=False)
        smiles = list(np.array(smiles)[sample_idx])
        if params["do_prop_pred"] and ("data_file" in params):
            if "reg_prop_tasks" in params:
                Y_reg = Y_reg[sample_idx]
            if "logit_prop_tasks" in params:
                Y_logit = Y_logit[sample_idx]

    print("Training set size is", len(smiles))
    print('first smiles: "', smiles[0], '"')
    print("total chars:", NCHARS)

    print("Vectorization...")
    X = mu.smiles_to_hot(smiles, MAX_LEN, params["PADDING"], CHAR_INDICES, NCHARS)

    print("Total Data size", X.shape[0])
    if np.shape(X)[0] % params["batch_size"] != 0:
        X = X[: np.shape(X)[0] // params["batch_size"] * params["batch_size"]]
        if params["do_prop_pred"]:
            if "reg_prop_tasks" in params:
                Y_reg = Y_reg[: np.shape(Y_reg)[0] // params["batch_size"] * params["batch_size"]]
            if "logit_prop_tasks" in params:
                Y_logit = Y_logit[: np.shape(Y_logit)[0] // params["batch_size"] * params["batch_size"]]

    np.random.seed(params["RAND_SEED"])
    rand_idx = np.arange(np.shape(X)[0])
    np.random.shuffle(rand_idx)

    val_split = 0  # params['val_split']
    TRAIN_FRAC = 1 - val_split
    num_train = int(X.shape[0] * TRAIN_FRAC)

    if num_train % params["batch_size"] != 0:
        num_train = num_train // params["batch_size"] * params["batch_size"]

    train_idx, test_idx = rand_idx[: int(num_train)], rand_idx[int(num_train) :]

    X_train, X_test = X[train_idx], X[test_idx]
    print("shape of input vector : {}", np.shape(X_train))
    print("Training set size is {}, after filtering to max length of {}".format(np.shape(X_train), MAX_LEN))

    if params["do_prop_pred"]:
        # !# add Y_train and Y_test here
        Y_train = []
        Y_test = []
        if "reg_prop_tasks" in params:
            Y_reg_train, Y_reg_test = Y_reg[train_idx], Y_reg[test_idx]
            Y_train.append(Y_reg_train)
            Y_test.append(Y_reg_test)
        if "logit_prop_tasks" in params:
            Y_logit_train, Y_logit_test = Y_logit[train_idx], Y_logit[test_idx]
            Y_train.append(Y_logit_train)
            Y_test.append(Y_logit_test)

        return X_train, X_test, Y_train, Y_test

    else:
        return X_train, X_test


def load_models(params):
    def identity(x):
        return K.identity(x)

    # def K_params with kl_loss_var
    kl_loss_var = K.variable(params["kl_loss_weight"])

    if params["reload_model"] == True:
        raise ValueError()
        encoder = load_encoder(params)
        decoder = load_decoder(params)
    else:
        encoder = encoder_model(params)
        decoder = decoder_model(params)

    x_in = encoder.inputs[0]

    z_mean, enc_output = encoder(x_in)
    z_samp, z_mean_log_var_output = variational_layers(z_mean, enc_output, kl_loss_var, params)

    # Decoder
    if params["do_tgru"]:
        x_out = decoder([z_samp, x_in])
    else:
        # raise ValueError()
        x_out = decoder(z_samp)

    x_out = Lambda(identity, name="x_pred")(x_out)
    model_outputs = [x_out, z_mean_log_var_output]

    AE_only_model = Model(x_in, model_outputs)

    if params["do_prop_pred"]:
        if params["reload_model"] == True:
            property_predictor = load_property_predictor(params)
        else:
            property_predictor = property_predictor_model(params)

        if (
            ("reg_prop_tasks" in params)
            and (len(params["reg_prop_tasks"]) > 0)
            and ("logit_prop_tasks" in params)
            and (len(params["logit_prop_tasks"]) > 0)
        ):
            reg_prop_pred, logit_prop_pred = property_predictor(z_mean)
            reg_prop_pred = Lambda(identity, name="reg_prop_pred")(reg_prop_pred)
            logit_prop_pred = Lambda(identity, name="logit_prop_pred")(logit_prop_pred)
            model_outputs.extend([reg_prop_pred, logit_prop_pred])

        # regression only scenario
        elif ("reg_prop_tasks" in params) and (len(params["reg_prop_tasks"]) > 0):
            reg_prop_pred = property_predictor(z_mean)
            reg_prop_pred = Lambda(identity, name="reg_prop_pred")(reg_prop_pred)
            model_outputs.append(reg_prop_pred)

        # logit only scenario
        elif ("logit_prop_tasks" in params) and (len(params["logit_prop_tasks"]) > 0):
            logit_prop_pred = property_predictor(z_mean)
            logit_prop_pred = Lambda(identity, name="logit_prop_pred")(logit_prop_pred)
            model_outputs.append(logit_prop_pred)

        else:
            raise ValueError("no logit tasks or regression tasks specified for property prediction")

        # making the models:
        AE_PP_model = Model(x_in, model_outputs)
        return (
            AE_only_model,
            AE_PP_model,
            encoder,
            decoder,
            property_predictor,
            kl_loss_var,
        )

    else:
        return AE_only_model, encoder, decoder, kl_loss_var


def kl_loss(truth_dummy, x_mean_log_var_output):
    x_mean, x_log_var = tf.split(x_mean_log_var_output, 2, axis=1)
    print("x_mean shape in kl_loss: ", x_mean.get_shape())
    kl_loss = -0.5 * K.mean(1 + x_log_var - K.square(x_mean) - K.exp(x_log_var), axis=-1)
    return kl_loss


def main_no_prop(seed, dataset):
    params = get_model_config("charvae", dataset)
    params["RAND_SEED"] = seed
    params = hyperparameters.load_params(params)
    start_time = time.time()
    output_dir = WB_LOG_DIR / "CHARVAE" / dataset / str(start_time)
    os.makedirs(output_dir, exist_ok=True)
    params["checkpoint_path"] = output_dir

    X_train, X_test = vectorize_data(params, dataset)
    AE_only_model, encoder, decoder, kl_loss_var = load_models(params)

    # compile models
    if params["optim"] == "adam":
        optim = Adam(lr=params["lr"], beta_1=params["momentum"])
    elif params["optim"] == "rmsprop":
        optim = RMSprop(lr=params["lr"], rho=params["momentum"])
    elif params["optim"] == "sgd":
        optim = SGD(lr=params["lr"], momentum=params["momentum"])
    else:
        raise NotImplemented("Please define valid optimizer")

    model_losses = {"x_pred": params["loss"], "z_mean_log_var": kl_loss}

    # vae metrics, callbacks
    vae_sig_schedule = partial(
        mol_cb.sigmoid_schedule,
        slope=params["anneal_sigmod_slope"],
        start=params["vae_annealer_start"],
    )
    vae_anneal_callback = mol_cb.WeightAnnealer_epoch(vae_sig_schedule, kl_loss_var, params["kl_loss_weight"], "vae")

    csv_clb = CSVLogger(output_dir / params["history_file"], append=False)
    callbacks = [vae_anneal_callback, csv_clb]
    callbacks.append(
        mol_cb.EncoderDecoderCheckpoint(
            encoder,
            decoder,
            params=params,
            prop_pred_model=False,
            save_best_only=False,
        )
    )

    def vae_anneal_metric(y_true, y_pred):
        return kl_loss_var

    xent_loss_weight = K.variable(params["xent_loss_weight"])
    model_train_targets = {
        "x_pred": X_train,
        "z_mean_log_var": np.ones((np.shape(X_train)[0], params["hidden_dim"] * 2)),
    }
    model_test_targets = {
        "x_pred": X_test,
        "z_mean_log_var": np.ones((np.shape(X_test)[0], params["hidden_dim"] * 2)),
    }

    AE_only_model.compile(
        loss=model_losses,
        loss_weights=[xent_loss_weight, kl_loss_var],
        optimizer=optim,
        metrics={"x_pred": ["categorical_accuracy", vae_anneal_metric]},
    )

    keras_verbose = params["verbose_print"]

    AE_only_model.fit(
        X_train,
        model_train_targets,
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        initial_epoch=params["prev_epochs"],
        callbacks=callbacks,
        verbose=keras_verbose,
        validation_data=[X_test, model_test_targets],
    )
    print("time of run : ", time.time() - start_time)
    print("**FINISHED**")
