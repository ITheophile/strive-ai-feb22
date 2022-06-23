from data_handler import df, train_df, test_df, train_loader, test_loader


print(df.shape)
print(len(train_df), len(test_df))
print(next(iter(train_loader))[0].shape)
print(next(iter(test_loader))[0].shape)
