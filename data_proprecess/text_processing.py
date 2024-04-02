import pandas as pd
from keybert import KeyBERT



def Get_key(doc):
    kw_model = KeyBERT(model="paraphrase-multilingual-MiniLM-L12-v2")
    keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 1),
                                         stop_words= ["的", "一", "在", "不", "了", ...] , use_mmr=True, diversity=0.7)
    print('keywords:', keywords)
    return keywords

def main():
    desc = pd.read_csv('../Preprocess_desc/title&desc.csv')
    desc['key'] = ''
    for i in range(desc.shape[0]):
        doc = str(desc.loc[i, 'description'])

        print(i)
        if doc != 'nan':
            keywords_with_scores = Get_key(doc)
            keywords = [phrase for phrase, score in keywords_with_scores]
            joined_keywords = ' '.join(keywords)
            print('doc:', doc)
            print('joined_keywords:', joined_keywords)
            desc.loc[i, 'key'] = str(desc.loc[i, 'title']) + ' ' + joined_keywords
        else:
            desc.loc[i, 'key'] = str(desc.loc[i, 'title'])
    desc.to_csv('../Preprocess_desc/keywords.csv')

def test():
    doc = '''
            选择您喜爱的英雄和职业并踏上一个永恒的地下城冒险。
    在冒险中随机获得技能与职业，创造一个属于您独一无二的战队吧。
    您究竟能不能战胜自己的极限呢？

    游戏特点
    1）Rogue lite游戏, 随机地遭遇敌人和特殊事件。
    2）探索地下城, 尽可能地深入探险这神秘的地下城。
    3）战略卡组构筑系统，从宝箱和敌人身上取得技能并加入自己的卡组来构筑属于你的独特卡组。
    4）角色扮演回合制战斗，游戏复杂但容易上手。挑战千奇百怪的敌人，有难度但令人上瘾。
    5）同时装备三个职业, 切换并使用其技能来施展强大的化学反应吧。
    6）使用职业和素材来合成不同独特的新职业。
    7）召唤您所喜爱的英雄吧！在地下城中被您击败的敌人将有一定几率从一个特别的召唤卡池里被召唤！
    8）收集不同的遗物来额外加强的构筑。
    9）游戏中含有很多各式各样的迷因, 动漫和电影元素！
    10）可购买以移除所有广告, 终身享有！
    11）能用一只手玩的游戏, 为什么要用两只手?
            '''
    keywords_with_scores = Get_key(doc)
    keywords = [phrase for phrase, score in keywords_with_scores]
    joined_keywords = ' '.join(keywords)
    print('doc:', doc)
    print('joined_keywords:', joined_keywords)


if __name__ == '__main__':
    main()
    # test()