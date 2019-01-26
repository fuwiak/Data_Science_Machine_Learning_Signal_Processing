#concatenate train and test set, to keep their indexes just dont use the ignore_index keyword
dataset =pd.concat([train_df, test_df], axis=0)
