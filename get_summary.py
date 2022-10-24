import numpy as np
from huggingface_hub import hf_hub_download
contribs = {
    "stereoset_all_bert_classifieronly": [-0.0025641024112701416, 0.010884352959692478, -0.001988487783819437, 0.0004709575732704252, -0.0016745161265134811, -0.0025641024112701416, -0.001831501955166459, 0.001988487783819437, -0.0017268446972593665, 0.0013082154328003526, -0.0005232862895354629, -0.005285191349685192, 0.0015175301814451814, -0.0004709576314780861, -0.0006279434892348945, 0.0029304029885679483, 0.002930403221398592, 0.0005232862313278019, 0.0035583467688411474, -5.232850162428804e-05, -0.0015175301814451814, 0.005285190884023905, 0.003924646880477667, 0.005442177411168814, 0.002459445269778371, 0.0021978020668029785, 0.001936158980242908, 0.011198326013982296, 0.0007849293178878725, 0.007639979477971792, 0.006698064040392637, -0.0016745161265134811, 0.00010465725790709257, -0.0041862898506224155, 0.005546834319829941, 0.002302459441125393, 0.008477237075567245, 9.313225884932663e-11, -0.0006802720599807799, 0.003924646880477667, 0.0011512298369780183, 0.0005232862895354629, -0.006959706544876099, 0.001098901266232133, -0.0009942438919097185, 0.004657247569411993, 0.002930403221398592, 0.0014128729235380888, 0.0008372580050490797, -0.004133961163461208, -0.008686551824212074, 0.009052853100001812, 0.004447932820767164, 0.0038199895061552525, 0.00177917315158993, 0.002721088472753763, 0.0028257458470761776, -0.0019884880166500807, 0.0020931449253112078, -0.0013082155492156744, -0.001988487783819437, 0.0011512297205626965, -0.008895866572856903, 0.0008895864593796432, 0.004918890539556742, 0.006384091917425394, 0.001988487783819437, 0.0007849294925108552, 0.021297750994563103, 0.0013605442363768816, 0.0009942438919097185, 0.004186290316283703, 1.1641532182693481e-10, 0.0025641026441007853, 0.0017268445808440447, 0.0023547881282866, 0.00036630031536333263, 0.011302981525659561, 0.002721088472753763, -0.0006802721763961017, 0.0041862898506224155, 0.0008895866922102869, 0.0, 0.003296703565865755, 0.002145473612472415, 0.0024594455026090145, 0.003192045958712697, 0.002197802299633622, 0.0026687600184231997, 0.001988487783819437, -0.0005232862313278019, 0.0006802720599807799, -0.0064887492917478085, 0.004971219692379236, 0.0016221873229369521, -0.0022501309867948294, 0.002459445269778371, 0.006750392261892557, 0.0010989010334014893, 0.0031397175043821335, 0.0028257458470761776, 0.006959706544876099, 0.004395604599267244, 0.003506017616018653, 0.013082155957818031, 0.010256409645080566, 0.02836211584508419, 0.0022501309867948294, 0.004290947690606117, 0.006017792504280806, 0.005023547913879156, 0.004500261973589659, 0.0022501307539641857, 0.006331763695925474, 0.004447932820767164, 0.0006802721763961017, 0.011773940175771713, 0.0017791733844205737, 0.011093668639659882, 0.02370486781001091, 0.006384091917425394, 0.0005232862895354629, 0.0016221872065216303, 0.010989010334014893, 0.0036106749903410673, -0.0004709575732704252, -0.001255886978469789, 0.03464154899120331, 0.005232862662523985, 0.004500261507928371, 0.02904239110648632, 0.01062270998954773, 0.011512297205626965, 0.0019884880166500807, 0.004447932820767164, 0.00962846726179123, 0.0019361593294888735, 0.0016221875557675958, 0.008791208267211914, -0.0006802719435654581, 0.01695447415113449, 0.0005232865223661065, -0.005494505632668734, -0.007535322103649378]
}
summary = dict()
for model, contrib in contribs.items():
    summary[model] = np.mean(np.absolute(contrib))

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

if __name__=="__main__":
    uniform_data = np.random.rand(10, 12)
    ax = sns.heatmap(np.array(contribs["stereoset_all_bert_classifieronly"]).reshape((12, 12)), linewidth=0.5)
    plt.show()

