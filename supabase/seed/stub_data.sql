-- 测试数据种子
insert into documents (content, metadata, embedding) values
  ('LangGraph 是一个用于构建有状态的多参与者应用程序的框架', '{"type": "concept", "lang": "zh"}', array_fill(0.1, array[1536])::vector),
  ('Supabase 提供了 PostgreSQL 数据库和向量搜索功能', '{"type": "concept", "lang": "zh"}', array_fill(0.2, array[1536])::vector),
  ('GraphRAG 结合了知识图谱和检索增强生成技术', '{"type": "concept", "lang": "zh"}', array_fill(0.3, array[1536])::vector);

insert into doc_relations (source_id, target_id, type, properties) values
  (1, 2, 'integrates_with', '{"weight": 0.9}'),
  (1, 3, 'implements', '{"weight": 0.95}'),
  (2, 3, 'enables', '{"weight": 0.85}');
