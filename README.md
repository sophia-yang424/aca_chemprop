# aca_chemprop
pki for a given molecule measures how well it binds to a specific receptor (For the specific dataset target is the Glucocorticoid receptor) and this moecule ace benchmark is trying to predict pki for any given molecule for a specfic receptor, with aca molecule (pairs) included in the dataset. the actual goal though is to ensur eour model is able to predict different/not simlar values for those aca pairs for the metric pki (who is not as important for our benchmakr, its just a metirc they chose)
However, it's worth noting that the MoleculeACE benchmark contains 30 different datasets (which you printed out earlier, like CHEMBL4203_Ki, CHEMBL287_Ki, etc.). Each of these corresponds to a different protein target or receptor. The benchmark tests if a model can handle activity cliffs across a wide variety of biological targets, not just one.

Cliff data: (small so i made dorpout range 0.1-0.6 to avoid overftting and also decreased hidden size search range, max is 300 because chemprops default hidden size is 300. a hidden size too big aka too many entires per hidden state (hidden state is a vector) will overfit
Full Data - Train: 508, Val: 90, Test: 152
Cliff Data - Train: 169, Val: 30, Test: 52
(intial chemprop benchmark used was sn2 barrier aka reaction activation energy. first used chemprop build it trainer which does hyperamater search to get optimal hypeeramters then I used those in  my own pipeline as sanity check. got about 0.94 R^2 using one of chemprops own benchmark datasets and the hyperparamters I found from doing hyperparameter search with their trainer which I then used on my own training pipeline, the params were max_lr, dropout, and a few others.we didnt even use optuna on that one, we just hdnaded in paramters we found from the other chemprop pipeline notebook i did)
(experiment with reaction based data)

Baseline chemprop model w/o loss modifcation (did optuna 50 trials to find otpimal hyperparamters. after we got those hyperparmters, optuna doesnt actually save your model th emodel it trains is temporary for space to search so many, so you need to train a model with pytorch using the paramters found):
(molecule based data so had to modify pipeline a little from the reaction based experiment)
mixed cliff data:
Loaded Best Model Checkpoint from Epoch: 20 (Val Loss: 0.4165)

>>> Tuned Model: Full CHEMBL2034_Ki Data Final Test Results <<<
Best Val Loss: 0.4165
Test R^2:      0.4005
Test MSE:      0.6049
Test RMSE:     0.7777
Test MAE:      0.5725
<img width="970" height="612" alt="image" src="https://github.com/user-attachments/assets/ca92b2ba-e98c-4059-83eb-3c2d12b61e66" />

(results are reproducible because i set determinstic to be true between runs with same seed and hyperparameters)
isolated cliff data:
Loaded Best Model Checkpoint from Epoch: 21 (Val Loss: 0.5093)

>>> Tuned Model: Activity Cliff Compounds Only Final Test Results <<<
Best Val Loss: 0.5093
Test R^2:      -0.0734
Test MSE:      0.9253
Test RMSE:     0.9619
Test MAE:      0.7769
<img width="962" height="595" alt="image" src="https://github.com/user-attachments/assets/80be504c-3f12-4ef7-830e-909972e6284e" />
(As expected, chemprop performs way worse on the activity cliff isolated set than the mixed one. It has a negative R^2 meaning the model fits doing worse than if you had a dummy model who predicts the average pki across all molecules, for each molecule. kinda like regression equiavelent of a dummy classifer who predicts one class every time)
------------------------
Notes:
A negative $R^2$$R^2$ means your model is doing worse than a simple mean predictor. If you just took the average of all the target values in your training set and guessed that exact average for every single test molecule, you would get an $R^2$$R^2$ of exactly 0.0. Getting a negative score means the model's predictions are so far off that just guessing the mean would have been more accurate!

so optuna builds an entirely separate arhcietcture (though the layout of it is same because we give it the same one each time to match chemprop arhceitcure of the mpnn + fnn setup) but with diff hyperpamaters each time like depth (number of layers in the mpnn aka number of iterations of the rnn in terms of chmeporops desing under hood) chemprops weight update is rnn based, how do we do that here?
answer: the chemprop "parts" like mpnn fnn are directly from chemprop module not us bulding from scratch in pytorch so it under hoood uses the rnn weight update stuff for the mpnn (chemprop bulit those parts under hood with pytorch too but we dont need to)
Because Chemprop is built entirely on PyTorch, all of its graph convolution layers, RNN-based message passing (BondMessagePassing), and feed-forward networks (RegressionFFN) in terms of all the layers that make up those blocks are just standard PyTorch nn.Modules already strung together/assembled by chemprop (we imported all those blocks from chemprop, not pytorch). By importing them directly, you get the absolute state-of-the-art graph neural network chemistry engine, but you retain the freedom to wrap it in your own PyTorch Lightning loop to mess with the loss function, metrics, and training dynamics.



model is still overfitting and hsoudlve stopped around epoch 11, how to make it do so since patience for dome reason didnt work
The reason the patience parameter didn't stop the training is likely due to micro-improvements. By default, PyTorch Lightning resets the early stopping patience counter even if the validation loss improves by a microscopic amount (like 0.00001). If the validation loss plateaus and just jitters slightly downwards, it won't trigger the stop.

We can fix this by introducing a min_delta parameter. This tells the early stopping mechanism: "If the loss doesn't improve by at least this much, consider it no improvement at all." without a min delta, itll only stop if you improve by less than 0 (which is impossible unless you literally do worse. it wont actually stop you from continuing if the improvements are minimal, only if you worsen). However, patience will be halted by micro improvements because patience counter RESETS any time we improve more than min_delta. so you could stagnate, improve a bit (some amount above min_delta), stagnate, then improve enough and just keep going because patience counts CONSECUTIVE non-imporvements (aka improvements < min_delta)

  checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1) (saves best val loss one so its ok if we go beyond a few epochs, itll still report metrics for this and save the params from those)
patience

q: strangely between runs despite having same hyperparams and same seed why is the perofmance diffferent
a: This is a very common issue when training Neural Networks (especially Graph Neural Networks) on GPUs.

Even though you used pl.seed_everything(42), PyTorch operations on the GPU are non-deterministic by default. Graph Neural Networks rely heavily on "scatter" and "aggregate" operations to pass messages between atoms. On a GPU, these operations use parallel threads that finish in an unpredictable order. Because of how floating-point math works, adding numbers in different orders produces tiny numerical differences. Over many epochs, these tiny differences compound, leading to different final metrics.


To enforce strict reproducibility, PyTorch Lightning requires passing deterministic=True to the Trainer. I've updated the training function to include this flag


separate optuna searcher objects (for diff search ranges) for the diff datasets (for exaple for the full ds, i chose lower range for dropout since the dataset wasnt as small as the ac only whod be very vulnerable to overfitting) but SAME trainer object for after we got optimal hyperparams from optuna and we train an actual persistent model (one for each ds to eval)
deterministic=True  # Forces deterministic GPU operations, its a paramter in the pytorch trainer
------------------------------------------
underfitting of baseline (no loss mods) for non cliff cases:
<img width="997" height="597" alt="image" src="https://github.com/user-attachments/assets/e8aca705-bac3-408e-ac5a-f0e2cf287f9f" />
something like this is obvious underiftting. but could you have it so where the training loss curve shape was same but we shift it higher? how would we know thats not ideal and not just bad data -> compare against a baseline of r^2 = 0 ( a dummy model who outputs only mean label across all data) and results from other hyperparamter choices
<img width="971" height="592" alt="image" src="https://github.com/user-attachments/assets/600cbd15-c71e-4cbb-bda7-d3be87aae10e" />
fix for underfitting: DECREASED lb of max_lr search range (to allow lower max_lr, youd think higher lr is better for underfitting, but pytorch uses noam training schedule so its a bit different. after decreasing the lower bound by about a factor of 10^1, it performed much better:
Performance for non cliff only, on baseline chemprop (no mods on loss function):
=== Best Hyperparameters for Non-Activity Cliff Compounds ===
Loaded Best Model Checkpoint from Epoch: 19 (Val Loss: 0.4968)
{'hidden_size': 179, 'depth': 3, 'dropout': 0.14325763295083002, 'init_lr': 1.0391097111495258e-05, 'max_lr': 0.0005189506922300196}
>>> Tuned Model: Non-Activity Cliff Compounds Final Test Results <<<
Best Val Loss: 0.4968
Test R^2:      0.4754
Test MSE:      0.5676
Test RMSE:     0.7534
Test MAE:      0.5899
<img width="1002" height="647" alt="image" src="https://github.com/user-attachments/assets/5866cc74-aa54-410f-b423-9b1d77bda468" />

