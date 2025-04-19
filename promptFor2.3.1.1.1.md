### 润色后的提示词（用于研究报告的 2.3.1.1.1 地域毗邻关系检索的知识网络可视化操作示例）：

以下是待处理CSV文档中的部分数据样例，采用“类型化边列表格式”存储。该数据共包含5列，其中最后一列表示边的类型，共有4种，分别为：  
1. **毗邻**  
2. **县、区级行政单位隶属于…市级行政单位**  
3. **分布地域**  
4. **特藏资源涉及乐种**  

数据列的具体含义如下：  
- **第1列**：源节点  
- **第2列**：源节点的类型  
- **第3列**：目标节点  
- **第4列**：目标节点的类型  
- **第5列**：边的类型  

基于该类型化边列表数据，若想绘制一个异构网络图，且用Python来实现，请帮我写出相应的代码
**注意**：（1）网络图中的箭头方向应从**源节点**指向**目标节点**（2）可能会用到networksx和matplotlib

```csv
sourceLabel,sourceNodeType,targetLabel,targetNodeType,relationType
石家庄市,places:City,阳泉市,places:City,毗邻
石家庄市,places:City,保定市,places:City,毗邻
邢台市,places:City,晋中市,places:City,毗邻
藁城区,places:County,石家庄市,places:City,县、区级行政单位隶属于…市级行政单位
宁晋县,places:County,邢台市,places:City,县、区级行政单位隶属于…市级行政单位
易县,places:County,保定市,places:City,县、区级行政单位隶属于…市级行政单位
冀中鼓吹乐-吹打班,ctm:MusicType,鹿泉区,places:County,分布地域
冀中鼓吹乐-吹打班,ctm:MusicType,深泽县,places:County,分布地域
河北音乐会-南乐会,ctm:MusicType,辛集市,places:County,分布地域
坠子戏,ctm:MusicType,深泽县,places:County,分布地域
河北省固安屈家营古乐资料集,ctm:SpecialIndependentResource,河北音乐会-北乐会,ctm:MusicType,特藏资源涉及乐种
河北省屈家营音乐会重建二十四周年（打击乐篇）：屈家营“音乐圣会”世界音乐周专场音乐会,ctm:SpecialIndependentResource,河北音乐会-北乐会,ctm:MusicType,特藏资源涉及乐种
"""2010春季学期艺术实践周——音乐学系晋中采风汇报""",ctm:SpecialIndependentResource,左权开花调,ctm:MusicType,特藏资源涉及乐种
古如歌,ctm:SpecialIndependentResource,鄂尔多斯古如歌,ctm:MusicType,特藏资源涉及乐种
```


---  

### 英文翻译：  

Below is a sample excerpt from the CSV document to be processed, which follows a "typed edge list format." The data consists of 5 columns, with the last column indicating the type of edge. There are 4 edge types in total:  
1. **Adjacent**  
2. **County/District-level administrative unit belongs to…City-level administrative unit**  
3. **Geographical distribution**  
4. **Special collection resources involve music genres**  

The columns are defined as follows:  
- **Column 1**: Source node  
- **Column 2**: Type of the source node  
- **Column 3**: Target node  
- **Column 4**: Type of the target node  
- **Column 5**: Type of the edge  

Based on this typed edge list data, to construct a heterogeneous network graph, using python, please help me by providing the corresponding code
**Note**:
 (1) The direction of the arrows in the graph should point from the **source node** to the **target node**;
 (2) You may refer to networksx and matplotlib
 (3) Presume we import the complete CSV file named "csv.csv" from the python script
 (4) For node labels, try preventing overlap by either:
    (4.1) Selectively displaying some labels, or
    (4.2) Replacing dense labels with numbered identifiers and including a separate legend that maps numbers to their corresponding labels
 (5) Enhance visualization with:
- Implement a force-directed (spring) layout model for optimal node positioning
- Scale node sizes proportionally to their centrality values for visual emphasis of important nodes