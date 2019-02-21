# Replace using median 
median = df['NUM_BEDROOMS'].median()
df['NUM_BEDROOMS'].fillna(median, inplace=True)

#concatenate train and test set, to keep their indexes just dont use the ignore_index keyword
dataset =pd.concat([train_df, test_df], axis=0)

#importance of columns
importances = list(zip(model.feature_importances_, df.columns))
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
