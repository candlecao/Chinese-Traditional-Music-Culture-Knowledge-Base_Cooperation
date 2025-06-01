# Chinese-Traditional-Music-Culture-Knowledge-Base_Cooperation

# 几次会议纪要
录制：彭黔平的快速会议
日期：2024-06-03 19:32:44
录制文件：https://meeting.tencent.com/v2/cloud-record/share?id=a3ecf5a0-465f-44cc-859d-8b8378a708c8&from=3&is-single=false&record_type=2
概要：
27'初步介绍了智能体配置：4 basic skills of an agent:
(1)has (or not) memory
(2)has (or not) expertise(using 提示词)
(3)has (or not) action ability(using 提示词)
(4)has (or not) ability of reflection(which demand more than one agents' coordination)
……后面提到很多technical issue, very unfamiliar to me

录制：彭黔平的快速会议
日期：2024-06-10 23:08:03
录制文件：https://meeting.tencent.com/v2/cloud-record/share?id=8ccf972d-52e0-431b-89b7-ee13de133479&from=3&is-single=false&record_type=2
Summary:
Use Knowledge Engine to do primary Knowledge Engineering on the .owl file, that is to recall ontology fragments according to the request of the users


录制：彭黔平预定的会议
日期：2024-06-27 23:45:39
录制文件：https://meeting.tencent.com/v2/cloud-record/share?id=b2dae294-4b34-4d45-90da-4f14596b238b&from=3&is-single=false&record_type=2
Summary:
角色设定；
任务介绍；
挂载资源（可以列出一系列资源）2'40"动态提示词，动态加载{}……分成两种，一种是用户可以随意改的，一种是系统的保留字（值），例如{Resources}
```
+ 资源
    以下是本体文件的资源信息
    {Resources} #类型 Session Value。彭：“{Resources}是我们自己定义的格式化的写法。不能改Resources这个命名，这是我们系统的关键词。”这里放的是文件信息，如我的本体文档的基本的元数据信息
    -----
    以下是一些与用户问题相关的知识图谱结构信息
    {refResources} #类型 Function Result（这个值是通过知识引擎去检索的）所以要选择一个FunctionCall（目前视频中是随便选的【根据问题检索资源】，因为彭老师还没录入Function……先告诉我怎么配，之后我自己配）。选参数：5'25"。18'05: 从知识仓库里召回的本体的结构片段
    -----
    这里加上：Context_GeneralIntroductionForCTM_Ontology_simplified.ttl中的一些片段
    -----
+ 任务要求（来自左侧的可选模块）
    任务要求：任务的SOP（即关于该怎么做这个任务？）
        独立完成任务
        直接输出任务指令（SPARQL语句），（我）不要任何多余的话
        尽量用中文或拼音
        你生成指令的时候，要基于每一个实体的类型设计合适的、准确的查询语句
        要给前缀bf写PREFIX
+ 自定义（来自左侧的可选模块）11'15"
    #任务的SOP
    //1.明确问题的核心……
    //2.识别相关标签和属性……
    //3.构建查询逻辑……
    ……
    ——用了LLM的COT的能力，即“思考链”
    #SPARQL例子
    <再给一个SPARQL的例子>
+ 连接插件（来自左侧的可选模块）13'25"
```
——动态提示词模板是比较复杂的，要根据用户的问题加载不同的报文结构，才能比较好地实现基于子图的预测——因为子图（根据上下文的context）是动态的eg:
limit:20,threshole:0.4

15'05"：怎么召回，用什么方法召回……我们自己的内置函数
20'50":智能体，用于连接各种各样的数据

22'10":大模型训练时的两种模式：一种是，改变部分的模型的参数（小参数的局部训练，即fine tuning，调优，会产生衍生的副作用，即“跷跷板效应”——即当优化了一个垂直领域的学习后，可能某些其他领域的效果会弱化）；另一种是全量的模型参数的变更，其代价非常大
25'45":既要兼顾垂直领域的灵活性，也要兼顾其专业知识性，即用RAG；如果垂直领域的灵活性是不重要的，可能用fine-tuning的效果更好
26'40":须解决模糊查询的问题：（27'25"）预处理，目前资源文件下面可被查询的标签有哪些，然后用模糊匹配的方式去匹配——对label不存在的异常情况做处置，建议让数据开发端团队来做此事（这个在技术上不难，而且很多数据库都有这个内生性的功能）
33'25":通过Chrome看前端指令