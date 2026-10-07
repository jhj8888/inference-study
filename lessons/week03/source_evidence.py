"""Rebuild line-numbered source anchors, requiring bytes from the full local lock."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT.parent/'vllm-main'
# id | file relative to vllm | qualified symbol | required literal within its source
SPECS = '''S01|entrypoints/openai/chat_completion/api_router.py|create_chat_completion|handler.create_chat_completion
S02|entrypoints/openai/chat_completion/serving.py|OpenAIServingChat._create_chat_completion|self.engine_client.generate
S03|entrypoints/openai/chat_completion/serving.py|OpenAIServingChat.render_chat_request|self.online_renderer.render_chat
S04|v1/engine/async_llm.py|AsyncLLM.generate|q = await self.add_request
S05|v1/engine/async_llm.py|AsyncLLM.add_request|self.input_processor.process_inputs
S06|v1/engine/async_llm.py|AsyncLLM._add_request|self.engine_core.add_request_async
S07|v1/engine/core_client.py|AsyncMPClient.add_request_async|EngineCoreRequestType.ADD
S08|v1/engine/core_client.py|AsyncMPClient._send_input_message|send_multipart
S09|v1/engine/core.py|EngineCoreProc.process_input_sockets|self.preprocess_add_request
S10|v1/engine/core.py|EngineCore.preprocess_add_request|Request.from_engine_core_request
S11|v1/engine/core.py|EngineCoreProc._handle_client_request|self.add_request
S12|v1/engine/core.py|EngineCore.add_request|self.scheduler.add_request
S13|v1/core/sched/scheduler.py|Scheduler.add_request|self._enqueue_waiting_request
S14|v1/core/sched/scheduler.py|Scheduler.schedule|self.kv_cache_manager.allocate_slots
S15|v1/core/sched/scheduler.py|Scheduler._update_after_schedule|request.num_computed_tokens +=
S16|v1/engine/core.py|EngineCore.step|self.scheduler.update_from_output
S17|v1/executor/uniproc_executor.py|UniProcExecutor.execute_model|self.collective_rpc
S18|v1/executor/uniproc_executor.py|UniProcExecutor.collective_rpc|run_method(self.driver_worker
S19|v1/worker/gpu_worker.py|Worker.execute_model|self.model_runner.execute_model
S20|v1/worker/gpu_model_runner.py|GPUModelRunner.execute_model|self._prepare_inputs
S21|v1/worker/gpu_model_runner.py|GPUModelRunner.sample_tokens|self.execute_model_state = None
S22|v1/core/sched/scheduler.py|Scheduler.update_from_output|self._free_request
S23|v1/core/sched/utils.py|check_stop|RequestStatus.FINISHED_LENGTH_CAPPED
S24|v1/engine/core.py|EngineCoreProc.process_output_sockets|send_multipart
S25|v1/engine/core_client.py|AsyncMPClient.get_output_async|outputs_queue.get
S26|v1/engine/async_llm.py|AsyncLLM._run_output_handler|output_processor.process_outputs
S27|v1/engine/output_processor.py|OutputProcessor.process_outputs|req_state.queue.put
S28|entrypoints/openai/chat_completion/serving.py|OpenAIServingChat.chat_completion_stream_generator|async for res in result_generator
S29|v1/engine/async_llm.py|AsyncLLM.abort|self.engine_core.abort_requests_async
S30|v1/engine/core.py|EngineCore.abort_requests|self.scheduler.finish_requests
S31|v1/core/sched/scheduler.py|Scheduler.finish_requests|self._free_request
S32|v1/core/sched/scheduler.py|Scheduler._free_request|self._free_blocks
S33|v1/core/sched/scheduler.py|Scheduler._free_request_blocks|self.kv_cache_manager.free
S34|v1/core/kv_cache_manager.py|KVCacheManager.free|self.coordinator.free
S35|v1/core/kv_cache_coordinator.py|KVCacheCoordinator.free|manager.free
S36|v1/core/single_type_kv_cache_manager.py|SingleTypeKVCacheManager.free|self.block_pool.free_blocks
S37|v1/core/block_pool.py|BlockPool.free_blocks|block.ref_cnt -= 1
S38|v1/core/sched/scheduler.py|Scheduler._preempt_request|self.waiting.prepend_request
S39|v1/core/kv_cache_manager.py|KVCacheManager.allocate_slots|if required_blocks > available_blocks
S40|v1/request.py|RequestStatus|FINISHED_ABORTED
S41|v1/core/sched/output.py|SchedulerOutput|num_scheduled_tokens: dict
S42|v1/engine/input_processor.py|InputProcessor.process_inputs|return EngineCoreRequest(
S43|entrypoints/serve/utils/api_utils.py|with_cancellation|handler_task
S44|v1/engine/core.py|EngineCoreProc._process_engine_step|self.step_fn
S45|v1/core/sched/scheduler.py|Scheduler._update_request_with_output|check_stop'''


def symbols(tree):
    result = {}
    def walk(node, prefix=''):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + child.name
                result[name] = child
                walk(child, name + '.')
            else: walk(child, prefix)
    walk(tree)
    return result


def build():
    lock = json.loads((ROOT/'sources/vllm-local-2026-10-07.lock.json').read_text(encoding='utf-8'))
    anchors = []
    for spec in SPECS.splitlines():
        key, name, symbol, needle = spec.split('|')
        name = 'vllm/' + name
        data = (SOURCE/name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == lock['files'][name]['sha256'], name
        text = data.decode('utf-8'); lines = text.splitlines()
        node = symbols(ast.parse(text))[symbol]
        matches = [i+1 for i in range(node.lineno-1,node.end_lineno) if needle in lines[i]]
        assert matches, (key, symbol, needle)
        anchors.append({'id':key,'file':name,'symbol':symbol,'start':node.lineno,'end':node.end_lineno,
                        'evidence_line':matches[0], 'evidence_text':lines[matches[0]-1].strip(),
                        'file_sha256':lock['files'][name]['sha256']})
    output = {'snapshot_identity':lock['identity'],'archive_sha256':lock['archive_sha256'],
              'method':'Static source inspection, not an engine execution trace.',
              'anchors':anchors}
    (HERE/'source-evidence.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(f'PASS: {len(anchors)} symbols and call/field anchors verified against full snapshot.')


if __name__ == '__main__': build()
