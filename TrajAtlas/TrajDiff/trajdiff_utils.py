from scipy.stats import binom


def _test_binom(length_df,
                times:int = 20):
    sumVal = length_df["true"] + length_df["false"]
    trueVal=length_df["true"]
    null=length_df["null"]
    p_val_list=[]
    for i in range(len(length_df)):
        if null[i]==0:
            null[i]=1/(sumVal[i]*times) # minimal
        p_val= 1- binom.cdf(trueVal[i], sumVal[i], null[i])
        if trueVal[i] == 0:
            p_val=1
        #p_val=1-poisson.cdf(rate[i], null[i])
        p_val_list.append(p_val)
    length_df["binom_p"]=p_val_list
    return(length_df)