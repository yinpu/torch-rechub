import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "torch-rechub",
  description: "A Lighting Pytorch Framework for Recommendation Models, Easy-to-use and Easy-to-extend.",
  head: [
    ['link', { rel: 'icon', href: '/torch-rechub/favicon.ico' }]
  ],

  base: '/torch-rechub/',

  markdown: {
    math: true
  },

  // 路径重写: 假设你的源文件都在 docs/en/ 下，但访问路径去掉 en
  rewrites: {
    'en/:rest*': ':rest*'
  },

  themeConfig: {
    logo: '/img/logo.png',
    search: { provider: 'local' },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/datawhalechina/torch-rechub' }
    ]
  },

  locales: {
    // ====================================================
    // 🇬🇧 English (Root)
    // ====================================================
    root: {
      label: 'English',
      lang: 'en',
      themeConfig: {
        nav: [
          { text: '🏠 Home', link: '/' },
          { text: '🚀 Getting Started', link: '/guide/intro' },
          { text: '⚙️ Core', link: '/core/intro' },
          { text: '🏰 Models', link: '/models/intro' },
          { text: '🛠️ Tools', link: '/tools/intro' },
          { text: '🚀 Serving', link: '/serving/intro' },
          { text: '📖 Tutorials', link: '/tutorials/intro' },
          { text: 'ℹ️ API', link: '/api/api' },
          { text: '👥 Community', link: '/community/faq' },
          { text: '📝 Blog', link: '/blog/match' }
        ],

        sidebar: {
          '/guide/': [
            {
              text: '🚀 Getting Started',
              items: [
                { text: 'Overview', link: '/guide/intro' },
                { text: 'Installation', link: '/guide/install' },
                { text: 'Quick Start', link: '/guide/quick_start' }
              ]
            }
          ],
          '/core/': [{
            text: '⚙️ Core Components', items: [
              { text: 'Overview', link: '/core/intro' },
              { text: 'Feature Columns', link: '/core/features' },
              { text: 'Data Pipeline', link: '/core/data' },
              { text: 'Training & Eval', link: '/core/evaluation' },
              { text: 'Trainer Hooks', link: '/core/trainer_hooks' }
            ]
          }],
          '/models/': [{
            text: '🏰 Model Zoo', items: [
              { text: 'Overview', link: '/models/intro' },
              { text: 'Ranking Models', link: '/models/ranking' },
              { text: 'Matching Models', link: '/models/matching' },
              { text: 'Multi-Task Models', link: '/models/mtl' },
              { text: 'Generative Models', link: '/models/generative' }
            ]
          }],
          '/tools/': [{
            text: '🛠️ Dev Tools', items: [
              { text: 'Overview', link: '/tools/intro' },
              { text: 'Visualization', link: '/tools/visualization' },
              { text: 'Experiment Tracking', link: '/tools/tracking' },
              { text: 'Callbacks', link: '/tools/callbacks' }
            ]
          }],
          '/serving/': [{
            text: '🚀 Serving', items: [
              { text: 'Overview', link: '/serving/intro' },
              { text: 'ONNX & Quantization', link: '/serving/onnx' },
              { text: 'Vector Indexing', link: '/serving/vector_index' },
              { text: 'Serving Demo', link: '/serving/demo' }
            ]
          }],
          '/tutorials/': [{
            text: '📖 Tutorials', items: [
              { text: 'Overview', link: '/tutorials/intro' },
              { text: 'CTR Pipeline', link: '/tutorials/ctr' },
              { text: 'Retrieval System', link: '/tutorials/retrieval' },
              { text: 'Big Data Pipeline', link: '/tutorials/pipeline' }
            ]
          }],

          '/api/': [
            {
              text: 'ℹ️ API Reference',
              items: [
                { text: 'Main API', link: '/api/api' },
              ]
            }
          ],
          '/community/': [
            {
              text: '📘 Community',
              items: [
                { text: 'FAQ', link: '/community/faq' },
                { text: 'Contributing', link: '/community/contributing' },
                { text: 'Changelog', link: '/community/changelog' }
              ]
            }
          ],
          '/blog/': [
            {
              text: '📝 Blog',
              items: [
                { text: 'Matching Models Guide', link: '/blog/match' },
                { text: 'Ranking Models Guide', link: '/blog/rank' },
                { text: 'HLLM Reproduction', link: '/blog/hllm_reproduction' }
              ]
            }
          ]
        }
      }
    },

    // ====================================================
    // 🇨🇳 中文 (Zh)
    // ====================================================
    zh: {
      label: '中文',
      lang: 'zh-CN',
      link: '/zh/',
      themeConfig: {
        nav: [
          { text: '🏠 首页', link: '/zh/' },
          { text: '🚀 快速入门', link: '/zh/guide/intro' },
          { text: '⚙️ 核心组件', link: '/zh/core/intro' },
          { text: '🏰 模型库', link: '/zh/models/intro' },
          { text: '🛠️ 研发工具', link: '/zh/tools/intro' },
          { text: '🚀 生产部署', link: '/zh/serving/intro' },
          { text: '📖 场景教程', link: '/zh/tutorials/intro' },
          { text: 'ℹ️ API', link: '/zh/api/api' },
          { text: '👥 社区', link: '/zh/community/faq' },
          { text: '📝 博客', link: '/zh/blog/match' }
        ],

        sidebar: {
          '/zh/guide/': [
            {
              text: '🚀 快速入门',
              items: [
                { text: '导览 (Overview)', link: '/zh/guide/intro' },
                { text: '安装指南', link: '/zh/guide/install' },
                { text: '3分钟上手', link: '/zh/guide/quick_start' }
              ]
            }
          ],
          '/zh/core/': [{
            text: '⚙️ 核心组件', items: [
              { text: '导览 (Overview)', link: '/zh/core/intro' },
              { text: '特征定义 (Features)', link: '/zh/core/features' },
              { text: '数据流水线 (Data)', link: '/zh/core/data' },
              { text: '训练与评估 (Eval)', link: '/zh/core/evaluation' },
              { text: 'Trainer Hook 自定义', link: '/zh/core/trainer_hooks' }
            ]
          }],
          '/zh/models/': [{
            text: '🏰 模型库', items: [
              { text: '导览 (Overview)', link: '/zh/models/intro' },
              { text: '排序模型 (Ranking)', link: '/zh/models/ranking' },
              { text: '召回模型 (Matching)', link: '/zh/models/matching' },
              { text: '多任务模型 (MTL)', link: '/zh/models/mtl' },
              { text: '生成式模型 (Generative)', link: '/zh/models/generative' }
            ]
          }],
          '/zh/tools/': [{
            text: '🛠️ 研发工具', items: [
              { text: '导览 (Overview)', link: '/zh/tools/intro' },
              { text: '可视化监控', link: '/zh/tools/visualization' },
              { text: '实验追踪', link: '/zh/tools/tracking' },
              { text: '回调函数', link: '/zh/tools/callbacks' }
            ]
          }],
          '/zh/serving/': [{
            text: '🚀 生产部署', items: [
              { text: '导览 (Overview)', link: '/zh/serving/intro' },
              { text: 'ONNX 导出与量化', link: '/zh/serving/onnx' },
              { text: '向量检索封装', link: '/zh/serving/vector_index' },
              { text: '在线服务示例', link: '/zh/serving/demo' }
            ]
          }],
          '/zh/tutorials/': [{
            text: '📖 场景教程', items: [
              { text: '导览 (Overview)', link: '/zh/tutorials/intro' },
              { text: 'CTR 预估流程', link: '/zh/tutorials/ctr' },
              { text: '召回系统搭建', link: '/zh/tutorials/retrieval' },
              { text: '全链路流水线', link: '/zh/tutorials/pipeline' }
            ]
          }],
          '/zh/api/': [
            {
              text: 'ℹ️ API Reference',
              items: [
                { text: 'API 参考', link: '/zh/api/api' },
              ]
            }
          ],
          '/zh/community/': [
            {
              text: '📘 社区信息',
              items: [
                { text: '常见问题 (FAQ)', link: '/zh/community/faq' },
                { text: '贡献指南 (Contributing)', link: '/zh/community/contributing' },
                { text: '版本日志 (Changelog)', link: '/zh/community/changelog' }
              ]
            }
          ],
          '/zh/blog/': [
            {
              text: '📝 博客',
              items: [
                { text: '召回模型训练指南', link: '/zh/blog/match' },
                { text: '排序模型训练指南', link: '/zh/blog/rank' },
                { text: 'HLLM 复现说明', link: '/zh/blog/hllm_reproduction' },
                { text: 'HSTU 复现说明', link: '/zh/blog/hstu_reproduction' }
              ]
            }
          ]
        }
      }
    }
  }
})
