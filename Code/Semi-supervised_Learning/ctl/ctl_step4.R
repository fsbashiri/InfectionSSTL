# Generate datasets for final model building, training, and testing

# use KLABELS and IMPUTED_LABELED datasets
df_training = read.csv(KLABELS)
df_all_labeled = read.csv(IMPUTED_LABELED)


labeled_encounteridVec = df_all_labeled$encounter_id
training_encounteridVec = df_training$encounter_id
validationVec = labeled_encounteridVec[which(labeled_encounteridVec %notin% training_encounteridVec)]
validation_df = df_all_labeled[which(df_all_labeled$encounter_id %in% validationVec),]

#TVAL
write.csv(validation_df, 'pykmeans-k4.20labeled_train_cluster_validate.csv', row.names = FALSE)
#TTRAIN
write.csv(df_training, 'pykmeans-k4.unlabeled_80labeled_train_cluster_infect_combo.csv', row.names = FALSE)
