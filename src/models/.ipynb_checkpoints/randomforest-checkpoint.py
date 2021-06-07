from sklearn.ensemble import RandomForestClassifier

def get_rf_model(n_estimators=100, max_depth=10):
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    return(clf)