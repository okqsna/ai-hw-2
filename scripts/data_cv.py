from sklearn.model_selection import StratifiedGroupKFold

def stratified_group_folds(frames, boxes, n_splits=3, seed=42):
    """
    Function creates cross-validation folds stratified by dominant class and grouped by video.

    :param frames: video frames
    """
    frames = frames.copy().reset_index(drop=True)

    objects = boxes.loc[boxes["iscrowd"].eq(0) & boxes["category_id"].between(1, 8) & boxes["image_id"].isin(frames["image_id"])]
    class_counts = (objects.groupby(["image_id", "category_id"]).size().unstack(fill_value=0).reindex(columns=range(1, 9), fill_value=0))
    frames["stratify_label"] = (frames["image_id"].map(class_counts.idxmax(axis=1)) .fillna(0).astype(int))

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, valid_idx) in enumerate(cv.split(frames, frames["stratify_label"], frames["video"])):
        frames[f"train_fold_{fold}"] = frames.index.isin(train_idx)
        frames.loc[valid_idx, "validation_fold"] = fold

    return frames