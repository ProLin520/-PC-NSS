# MultiSource_DOA

`MultiSource_DOA` 是一个独立的多源 DOA 研究项目，研究少快拍、强相关/相干、近间隔双源条件下的可解释深度学习增强传统子空间算法。

当前批准主线是 **多尺度、结构保持、分辨率感知的 PC-NSS**：网络融合多个子阵尺度的前后向空间平滑物理证据，输出受限的 lag 域协方差修正，经结构投影后由固定 Root-MUSIC 产生最终 DOA。

本项目与同级 `DIO_DOA` 相互独立。`DIO_DOA` 保留单源 PALR、门控失败和负结果；新项目不直接导入或修改它的运行代码。

当前状态：设计规格已形成，模型代码尚未实现。设计见 `docs/superpowers/specs/2026-08-04-multiscale-pcnss-design.md`。
