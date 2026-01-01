"""
Tests for Enterprise Agent (THINK -> PLAN -> APPROVE -> EXECUTE -> VALIDATE -> REFLECT).

Module: subagent-manager/tests/test_enterprise_agent.py

Tests governance, provenance tracking, and audit logging modules.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
import sys
from pathlib import Path

# Setup path for imports
service_path = Path(__file__).parent.parent / "service"
sys.path.insert(0, str(service_path.parent))
sys.path.insert(0, str(service_path))

# Import individual modules that don't have relative import dependencies
import importlib.util


def load_module(name: str, path: Path):
    """Load a module from path, avoiding relative import issues."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load modules in dependency order
governance_module = load_module("governance", service_path / "governance.py")
provenance_module = load_module("provenance", service_path / "provenance.py")
audit_logger_module = load_module("audit_logger", service_path / "audit_logger.py")

# Import from loaded modules
GovernanceGate = governance_module.GovernanceGate
ApprovalResult = governance_module.ApprovalResult
PolicyViolation = governance_module.PolicyViolation
GovernancePolicy = governance_module.GovernancePolicy
PolicyCategory = governance_module.PolicyCategory
PolicySeverity = governance_module.PolicySeverity
HumanApprovalQueue = governance_module.HumanApprovalQueue

ProvenanceTracker = provenance_module.ProvenanceTracker
ProvenanceRecord = provenance_module.ProvenanceRecord
ProvenanceChain = provenance_module.ProvenanceChain
Artifact = provenance_module.Artifact
ActorType = provenance_module.ActorType
ArtifactType = provenance_module.ArtifactType

AuditLogger = audit_logger_module.AuditLogger
AuditEvent = audit_logger_module.AuditEvent
AuditPhase = audit_logger_module.AuditPhase
AuditSeverity = audit_logger_module.AuditSeverity
MemoryAuditSink = audit_logger_module.MemoryAuditSink
ConsoleAuditSink = audit_logger_module.ConsoleAuditSink


# ============================================================================
# Governance Tests
# ============================================================================


class TestGovernanceGate:
    """Tests for GovernanceGate."""

    def test_default_policies(self) -> None:
        """Test default policy creation."""
        gate = GovernanceGate()

        assert len(gate.policies) > 0
        assert any(p.name == "high_risk_tool_execution" for p in gate.policies)
        assert any(p.name == "file_system_write" for p in gate.policies)

    def test_high_risk_tools(self) -> None:
        """Test high risk tool identification."""
        gate = GovernanceGate()

        assert "bash_exec" in gate.high_risk_tools
        assert "file_delete" in gate.high_risk_tools
        assert "deploy" in gate.high_risk_tools

    @pytest.mark.asyncio
    async def test_approve_low_risk_plan(self) -> None:
        """Test approval of low-risk plan."""
        gate = GovernanceGate()

        # Create a low-risk plan
        plan = MagicMock()
        plan.plan_id = "test-plan-123"
        plan.risk_level = "low"
        plan.steps = [
            MagicMock(tool="file_read", action="Read file"),
            MagicMock(tool="file_glob", action="Search files"),
        ]
        plan.model_dump.return_value = {"plan_id": plan.plan_id}

        config = MagicMock()
        config.auto_approve_low_risk = True
        config.max_risk_level_auto_approve = "low"

        result = await gate.evaluate(
            plan=plan,
            context={"user_id": "test-user"},
            config=config,
        )

        assert result.approved is True
        assert result.approver == "auto_policy"

    @pytest.mark.asyncio
    async def test_deny_high_risk_plan(self) -> None:
        """Test denial of high-risk plan."""
        gate = GovernanceGate()

        # Create a high-risk plan
        plan = MagicMock()
        plan.plan_id = "test-plan-456"
        plan.risk_level = "critical"
        plan.steps = [
            MagicMock(tool="bash_exec", action="Run command"),
            MagicMock(tool="deploy", action="Deploy to prod"),
        ]
        plan.model_dump.return_value = {"plan_id": plan.plan_id}

        config = MagicMock()
        config.auto_approve_low_risk = True
        config.max_risk_level_auto_approve = "low"

        result = await gate.evaluate(
            plan=plan,
            context={"user_id": "test-user"},
            config=config,
        )

        # Critical risk requires human approval
        assert result.approved is False
        assert result.requires_human is True
        assert len(result.violations) > 0

    def test_add_custom_policy(self) -> None:
        """Test adding custom policy."""
        gate = GovernanceGate()
        initial_count = len(gate.policies)

        custom_policy = GovernancePolicy(
            name="custom_api_restriction",
            category=PolicyCategory.EXTERNAL_SERVICES,
            description="Custom API call restriction",
            trigger_tools={"custom_api"},
            severity=PolicySeverity.WARNING,
        )

        gate.add_policy(custom_policy)

        assert len(gate.policies) == initial_count + 1
        assert any(p.name == "custom_api_restriction" for p in gate.policies)

    def test_remove_policy(self) -> None:
        """Test removing a policy."""
        gate = GovernanceGate()

        # Add then remove
        custom_policy = GovernancePolicy(
            name="temp_policy",
            category=PolicyCategory.COMPLIANCE,
            description="Temporary policy",
        )
        gate.add_policy(custom_policy)

        removed = gate.remove_policy("temp_policy")
        assert removed is True
        assert not any(p.name == "temp_policy" for p in gate.policies)

    @pytest.mark.asyncio
    async def test_sensitive_pattern_detection(self) -> None:
        """Test detection of sensitive data patterns."""
        gate = GovernanceGate()

        # Plan with sensitive pattern
        plan = MagicMock()
        plan.plan_id = "sensitive-plan"
        plan.risk_level = "low"
        plan.steps = [MagicMock(tool="file_read", action="read")]
        plan.model_dump.return_value = {"api_key": "secret123"}

        config = MagicMock()
        config.auto_approve_low_risk = True
        config.max_risk_level_auto_approve = "low"

        result = await gate.evaluate(plan=plan, context={}, config=config)

        # Should detect sensitive pattern
        assert result.approved is False
        assert result.requires_human is True


class TestPolicyViolation:
    """Tests for PolicyViolation model."""

    def test_violation_creation(self) -> None:
        """Test creating a policy violation."""
        violation = PolicyViolation(
            policy_name="test_policy",
            category=PolicyCategory.CODE_EXECUTION,
            severity=PolicySeverity.ERROR,
            description="Test violation",
            affected_tool="bash_exec",
            remediation="Review command",
        )

        assert violation.policy_name == "test_policy"
        assert violation.severity == PolicySeverity.ERROR
        assert violation.affected_tool == "bash_exec"
        assert violation.violation_id is not None

    def test_violation_timestamp(self) -> None:
        """Test violation has timestamp."""
        violation = PolicyViolation(
            policy_name="test",
            category=PolicyCategory.DATA_ACCESS,
            description="Test",
        )

        assert violation.timestamp is not None
        assert isinstance(violation.timestamp, datetime)


class TestHumanApprovalQueue:
    """Tests for HumanApprovalQueue."""

    @pytest.mark.asyncio
    async def test_request_approval(self) -> None:
        """Test requesting human approval."""
        queue = HumanApprovalQueue()

        plan = MagicMock()
        violations = [PolicyViolation(
            policy_name="test",
            category=PolicyCategory.CODE_EXECUTION,
            description="Test",
        )]

        request_id = await queue.request_approval(
            plan=plan,
            violations=violations,
            context={"user_id": "test"},
        )

        assert request_id is not None
        assert request_id in queue._pending

    @pytest.mark.asyncio
    async def test_approve_request(self) -> None:
        """Test approving a pending request."""
        queue = HumanApprovalQueue()

        request_id = await queue.request_approval(
            plan=MagicMock(),
            violations=[],
            context={},
        )

        result = await queue.approve(
            request_id=request_id,
            approver_id="admin-user",
            conditions=["Monitor execution"],
        )

        assert result.approved is True
        assert result.approver == "admin-user"
        assert "Monitor execution" in result.conditions

    @pytest.mark.asyncio
    async def test_deny_request(self) -> None:
        """Test denying a pending request."""
        queue = HumanApprovalQueue()

        request_id = await queue.request_approval(
            plan=MagicMock(),
            violations=[],
            context={},
        )

        result = await queue.deny(
            request_id=request_id,
            approver_id="admin-user",
            reason="Too risky",
        )

        assert result.approved is False
        assert "Too risky" in result.reason

    @pytest.mark.asyncio
    async def test_check_pending_status(self) -> None:
        """Test checking status of pending request."""
        queue = HumanApprovalQueue()

        request_id = await queue.request_approval(
            plan=MagicMock(),
            violations=[],
            context={},
        )

        status = await queue.check_status(request_id)
        assert status is None  # Still pending

    @pytest.mark.asyncio
    async def test_check_completed_status(self) -> None:
        """Test checking status of completed request."""
        queue = HumanApprovalQueue()

        request_id = await queue.request_approval(
            plan=MagicMock(),
            violations=[],
            context={},
        )

        await queue.approve(request_id=request_id, approver_id="admin")

        status = await queue.check_status(request_id)
        assert status is not None
        assert status.approved is True


# ============================================================================
# Provenance Tests
# ============================================================================


class TestProvenanceTracker:
    """Tests for ProvenanceTracker."""

    def test_hash_data_string(self) -> None:
        """Test hashing string data."""
        tracker = ProvenanceTracker()

        hash1 = tracker.hash_data("test data")
        hash2 = tracker.hash_data("test data")
        hash3 = tracker.hash_data("different data")

        assert hash1 == hash2  # Same input = same hash
        assert hash1 != hash3  # Different input = different hash
        assert len(hash1) == 64  # SHA256 = 64 hex chars

    def test_hash_data_dict(self) -> None:
        """Test hashing dict data."""
        tracker = ProvenanceTracker()

        data = {"key": "value", "number": 42}
        hash1 = tracker.hash_data(data)
        hash2 = tracker.hash_data({"number": 42, "key": "value"})  # Different order

        assert hash1 == hash2  # JSON sorted keys = same hash

    def test_hash_data_none(self) -> None:
        """Test hashing None."""
        tracker = ProvenanceTracker()

        hash_none = tracker.hash_data(None)
        assert hash_none is not None
        assert len(hash_none) == 64

    def test_hash_data_bytes(self) -> None:
        """Test hashing bytes."""
        tracker = ProvenanceTracker()

        hash1 = tracker.hash_data(b"test bytes")
        hash2 = tracker.hash_data(b"test bytes")

        assert hash1 == hash2

    def test_record_provenance(self) -> None:
        """Test recording provenance."""
        tracker = ProvenanceTracker()

        record = tracker.record(
            actor_id="agent-123",
            actor_type=ActorType.ENTERPRISE_AGENT,
            action="read_file",
            inputs_hash="abc123",
            outputs_hash="def456",
            tool_id="file_read",
        )

        assert record.provenance_id is not None
        assert record.actor_id == "agent-123"
        assert record.tool_id == "file_read"
        assert record.inputs_hash == "abc123"

        # Verify stored
        stored = tracker.get_record(record.provenance_id)
        assert stored is not None
        assert stored.provenance_id == record.provenance_id

    def test_record_with_string_actor_type(self) -> None:
        """Test recording with string actor type."""
        tracker = ProvenanceTracker()

        record = tracker.record(
            actor_id="agent-123",
            actor_type="enterprise_agent",  # String instead of enum
            action="test",
            inputs_hash="a",
            outputs_hash="b",
        )

        assert record.actor_type == ActorType.ENTERPRISE_AGENT

    def test_provenance_chain(self) -> None:
        """Test provenance chain management."""
        tracker = ProvenanceTracker()

        # Start chain
        chain = tracker.start_chain(execution_id="exec-123")
        assert chain.execution_id == "exec-123"
        assert len(chain.records) == 0

        # Add records
        tracker.record(
            actor_id="agent",
            actor_type=ActorType.ENTERPRISE_AGENT,
            action="step_1",
            inputs_hash="a",
            outputs_hash="b",
        )
        tracker.record(
            actor_id="agent",
            actor_type=ActorType.ENTERPRISE_AGENT,
            action="step_2",
            inputs_hash="c",
            outputs_hash="d",
        )

        # Finalize
        finalized = tracker.finalize_chain(chain.chain_id)

        assert finalized.finalized_at is not None
        assert finalized.root_hash != ""
        assert len(finalized.records) == 2

    def test_verify_chain(self) -> None:
        """Test chain verification."""
        tracker = ProvenanceTracker()

        chain = tracker.start_chain(execution_id="exec-456")
        tracker.record(
            actor_id="agent",
            actor_type=ActorType.ENTERPRISE_AGENT,
            action="action",
            inputs_hash="x",
            outputs_hash="y",
        )
        tracker.finalize_chain(chain.chain_id)

        # Verify
        assert tracker.verify_chain(chain.chain_id) is True

    def test_verify_unfinalized_chain(self) -> None:
        """Test verifying unfinalized chain."""
        tracker = ProvenanceTracker()

        chain = tracker.start_chain(execution_id="unfinalized")
        tracker.record(
            actor_id="agent",
            actor_type=ActorType.ENTERPRISE_AGENT,
            action="action",
            inputs_hash="x",
            outputs_hash="y",
        )

        # Should return True for unfinalized chain
        assert tracker.verify_chain(chain.chain_id) is True

    def test_create_artifact(self) -> None:
        """Test artifact creation with provenance."""
        tracker = ProvenanceTracker()

        artifact = tracker.create_artifact(
            artifact_type=ArtifactType.EXECUTION_PLAN,
            content={"plan": "test plan"},
            created_by="agent-123",
            provenance_id="prov-123",
        )

        assert artifact.artifact_id is not None
        assert artifact.content_hash != ""
        assert artifact.created_by == "agent-123"

        # Verify
        assert tracker.verify_artifact(artifact) is True

    def test_export_import_chain(self) -> None:
        """Test exporting and importing chains."""
        tracker = ProvenanceTracker()

        # Create chain with records
        chain = tracker.start_chain(execution_id="export-test")
        tracker.record(
            actor_id="agent",
            actor_type=ActorType.ENTERPRISE_AGENT,
            action="test",
            inputs_hash="1",
            outputs_hash="2",
        )
        tracker.finalize_chain(chain.chain_id)

        # Export
        exported = tracker.export_chain(chain.chain_id)
        assert "chain_id" in exported
        assert "records" in exported
        assert exported["record_count"] == 1

        # Import into new tracker
        new_tracker = ProvenanceTracker()
        imported = new_tracker.import_chain(exported)

        assert imported.chain_id == chain.chain_id
        assert len(imported.records) == 1

    def test_get_records_by_actor(self) -> None:
        """Test getting records by actor."""
        tracker = ProvenanceTracker()

        tracker.record(
            actor_id="agent-1",
            actor_type=ActorType.ENTERPRISE_AGENT,
            action="action1",
            inputs_hash="a",
            outputs_hash="b",
        )
        tracker.record(
            actor_id="agent-2",
            actor_type=ActorType.ENTERPRISE_AGENT,
            action="action2",
            inputs_hash="c",
            outputs_hash="d",
        )
        tracker.record(
            actor_id="agent-1",
            actor_type=ActorType.ENTERPRISE_AGENT,
            action="action3",
            inputs_hash="e",
            outputs_hash="f",
        )

        records = tracker.get_records_by_actor("agent-1")
        assert len(records) == 2

    def test_get_statistics(self) -> None:
        """Test getting tracker statistics."""
        tracker = ProvenanceTracker()

        tracker.record(
            actor_id="agent",
            actor_type=ActorType.ENTERPRISE_AGENT,
            action="action",
            inputs_hash="a",
            outputs_hash="b",
            tool_id="file_read",
        )

        stats = tracker.get_statistics()

        assert stats["total_records"] == 1
        assert stats["records_by_actor_type"]["enterprise_agent"] == 1
        assert stats["records_by_tool"]["file_read"] == 1


# ============================================================================
# Audit Logger Tests
# ============================================================================


class TestAuditLogger:
    """Tests for AuditLogger."""

    def test_log_event(self) -> None:
        """Test logging an audit event."""
        sink = MemoryAuditSink()
        logger = AuditLogger(sinks=[sink])

        event = AuditEvent(
            phase=AuditPhase.START,
            execution_id="exec-123",
            task="Test task",
            user_id="user-456",
        )

        event_id = logger.log_event(event)

        assert event_id is not None
        assert len(sink.entries) == 1
        assert sink.entries[0].event.phase == AuditPhase.START

    def test_log_all_phases(self) -> None:
        """Test logging events for all phases."""
        sink = MemoryAuditSink()
        logger = AuditLogger(sinks=[sink])

        phases = [
            AuditPhase.START,
            AuditPhase.THINK,
            AuditPhase.PLAN,
            AuditPhase.APPROVE,
            AuditPhase.EXECUTE,
            AuditPhase.VALIDATE,
            AuditPhase.REFLECT,
            AuditPhase.COMPLETE,
        ]

        for phase in phases:
            logger.log_event(AuditEvent(
                phase=phase,
                execution_id="exec-123",
            ))

        assert len(sink.entries) == 8
        stats = logger.get_statistics()
        assert stats["total_events"] == 8

    def test_query_by_execution(self) -> None:
        """Test querying events by execution ID."""
        sink = MemoryAuditSink()
        logger = AuditLogger(sinks=[sink])

        logger.log_event(AuditEvent(phase=AuditPhase.START, execution_id="exec-1"))
        logger.log_event(AuditEvent(phase=AuditPhase.START, execution_id="exec-2"))
        logger.log_event(AuditEvent(phase=AuditPhase.PLAN, execution_id="exec-1"))

        exec1_events = sink.get_entries(execution_id="exec-1")
        assert len(exec1_events) == 2

        exec2_events = sink.get_entries(execution_id="exec-2")
        assert len(exec2_events) == 1

    def test_query_by_phase(self) -> None:
        """Test querying events by phase."""
        sink = MemoryAuditSink()
        logger = AuditLogger(sinks=[sink])

        logger.log_event(AuditEvent(phase=AuditPhase.START, execution_id="exec-1"))
        logger.log_event(AuditEvent(phase=AuditPhase.PLAN, execution_id="exec-1"))
        logger.log_event(AuditEvent(phase=AuditPhase.PLAN, execution_id="exec-2"))

        plan_events = sink.get_entries(phase=AuditPhase.PLAN)
        assert len(plan_events) == 2

    def test_redaction(self) -> None:
        """Test sensitive data redaction."""
        sink = MemoryAuditSink()
        logger = AuditLogger(sinks=[sink], redact_sensitive=True)

        event = AuditEvent(
            phase=AuditPhase.START,
            task="api_key=secret123 password=mypass",
        )

        logger.log_event(event)

        logged_event = sink.entries[0].event
        assert "secret123" not in logged_event.task
        assert "mypass" not in logged_event.task
        assert "REDACTED" in logged_event.task

    def test_no_redaction(self) -> None:
        """Test with redaction disabled."""
        sink = MemoryAuditSink()
        logger = AuditLogger(sinks=[sink], redact_sensitive=False)

        event = AuditEvent(
            phase=AuditPhase.START,
            task="api_key=secret123",
        )

        logger.log_event(event)

        logged_event = sink.entries[0].event
        assert "secret123" in logged_event.task

    def test_event_hooks(self) -> None:
        """Test event hooks."""
        sink = MemoryAuditSink()
        logger = AuditLogger(sinks=[sink])

        hook_calls = []

        def test_hook(event: AuditEvent) -> None:
            hook_calls.append(event.phase)

        logger.add_hook(test_hook)

        logger.log_event(AuditEvent(phase=AuditPhase.START))
        logger.log_event(AuditEvent(phase=AuditPhase.PLAN))

        assert len(hook_calls) == 2
        assert AuditPhase.START in hook_calls
        assert AuditPhase.PLAN in hook_calls

    def test_remove_hook(self) -> None:
        """Test removing a hook."""
        sink = MemoryAuditSink()
        logger = AuditLogger(sinks=[sink])

        hook_calls = []

        def test_hook(event: AuditEvent) -> None:
            hook_calls.append(event.phase)

        logger.add_hook(test_hook)
        logger.log_event(AuditEvent(phase=AuditPhase.START))

        logger.remove_hook(test_hook)
        logger.log_event(AuditEvent(phase=AuditPhase.PLAN))

        assert len(hook_calls) == 1  # Only first event

    def test_multiple_sinks(self) -> None:
        """Test logging to multiple sinks."""
        sink1 = MemoryAuditSink()
        sink2 = MemoryAuditSink()
        logger = AuditLogger(sinks=[sink1, sink2])

        logger.log_event(AuditEvent(phase=AuditPhase.START))

        assert len(sink1.entries) == 1
        assert len(sink2.entries) == 1

    def test_get_events_by_execution(self) -> None:
        """Test getting events by execution from logger."""
        sink = MemoryAuditSink()
        logger = AuditLogger(sinks=[sink])

        logger.log_event(AuditEvent(phase=AuditPhase.START, execution_id="test-exec"))
        logger.log_event(AuditEvent(phase=AuditPhase.PLAN, execution_id="test-exec"))

        events = logger.get_events_by_execution("test-exec")
        assert len(events) == 2


class TestAuditEvent:
    """Tests for AuditEvent model."""

    def test_phase_specific_fields(self) -> None:
        """Test phase-specific fields."""
        # PLAN phase
        plan_event = AuditEvent(
            phase=AuditPhase.PLAN,
            plan_id="plan-123",
            confidence=0.85,
            risk_level="medium",
            steps_count=5,
        )
        assert plan_event.plan_id == "plan-123"
        assert plan_event.confidence == 0.85

        # EXECUTE phase
        exec_event = AuditEvent(
            phase=AuditPhase.EXECUTE,
            steps_completed=4,
            steps_failed=1,
            success_rate=0.8,
        )
        assert exec_event.steps_completed == 4
        assert exec_event.success_rate == 0.8

        # COMPLETE phase
        complete_event = AuditEvent(
            phase=AuditPhase.COMPLETE,
            success=True,
            iterations_used=2,
            total_duration_ms=5000.0,
        )
        assert complete_event.success is True
        assert complete_event.total_duration_ms == 5000.0

    def test_error_phase(self) -> None:
        """Test error phase fields."""
        error_event = AuditEvent(
            phase=AuditPhase.ERROR,
            error="Something went wrong",
            error_type="RuntimeError",
        )

        assert error_event.error == "Something went wrong"
        assert error_event.error_type == "RuntimeError"

    def test_tool_call_phase(self) -> None:
        """Test tool call phase fields."""
        tool_event = AuditEvent(
            phase=AuditPhase.TOOL_CALL,
            tool_name="file_read",
            tool_duration_ms=150.5,
            tool_inputs_hash="abc123",
            tool_outputs_hash="def456",
        )

        assert tool_event.tool_name == "file_read"
        assert tool_event.tool_duration_ms == 150.5


class TestMemoryAuditSink:
    """Tests for MemoryAuditSink."""

    def test_max_entries(self) -> None:
        """Test max entries limit."""
        sink = MemoryAuditSink(max_entries=5)

        # Add 10 entries
        for i in range(10):
            from audit_logger import AuditLogEntry
            entry = AuditLogEntry(
                event=AuditEvent(phase=AuditPhase.START),
                formatted_message=f"Event {i}",
            )
            import asyncio
            asyncio.get_event_loop().run_until_complete(sink.write(entry))

        assert len(sink.entries) == 5  # Limited to max

    def test_clear(self) -> None:
        """Test clearing entries."""
        sink = MemoryAuditSink()

        from audit_logger import AuditLogEntry
        entry = AuditLogEntry(
            event=AuditEvent(phase=AuditPhase.START),
            formatted_message="Test",
        )
        import asyncio
        asyncio.get_event_loop().run_until_complete(sink.write(entry))

        assert len(sink.entries) == 1

        sink.clear()
        assert len(sink.entries) == 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestEnterpriseIntegration:
    """Integration tests for enterprise agent components."""

    @pytest.mark.asyncio
    async def test_full_audit_trail(self) -> None:
        """Test that full audit trail is created."""
        sink = MemoryAuditSink()
        audit = AuditLogger(sinks=[sink])
        provenance = ProvenanceTracker()

        execution_id = "integration-test-123"

        # Start chain
        chain = provenance.start_chain(execution_id=execution_id)

        # Log start - write directly to sink for sync behavior
        from audit_logger import AuditLogEntry
        event1 = AuditEvent(
            phase=AuditPhase.START,
            execution_id=execution_id,
            task="Integration test",
        )
        entry1 = AuditLogEntry(event=event1, formatted_message="Start")
        await sink.write(entry1)

        # Record action
        record = provenance.record(
            actor_id=execution_id,
            actor_type=ActorType.ENTERPRISE_AGENT,
            action="test_action",
            inputs_hash=provenance.hash_data({"input": "test"}),
            outputs_hash=provenance.hash_data({"output": "result"}),
        )

        # Log complete - write directly
        event2 = AuditEvent(
            phase=AuditPhase.COMPLETE,
            execution_id=execution_id,
            success=True,
            provenance_chain=[record.provenance_id],
        )
        entry2 = AuditLogEntry(event=event2, formatted_message="Complete")
        await sink.write(entry2)

        # Finalize chain
        provenance.finalize_chain(chain.chain_id)

        # Verify
        events = sink.get_entries(execution_id=execution_id)
        assert len(events) == 2

        assert provenance.verify_chain(chain.chain_id) is True

        exported = provenance.export_chain(chain.chain_id)
        assert exported["record_count"] == 1

    @pytest.mark.asyncio
    async def test_governance_with_audit(self) -> None:
        """Test governance gate with audit logging."""
        sink = MemoryAuditSink()
        gate = GovernanceGate()

        # Create plan
        plan = MagicMock()
        plan.plan_id = "governed-plan"
        plan.risk_level = "low"
        plan.steps = [MagicMock(tool="file_read", action="read")]
        plan.model_dump.return_value = {"plan_id": plan.plan_id}

        config = MagicMock()
        config.auto_approve_low_risk = True
        config.max_risk_level_auto_approve = "low"

        # Evaluate
        result = await gate.evaluate(plan=plan, context={}, config=config)

        # Log result - write directly to sink
        from audit_logger import AuditLogEntry
        event = AuditEvent(
            phase=AuditPhase.APPROVE,
            execution_id="test",
            approved=result.approved,
            approver=result.approver,
            violations=[v.description for v in result.violations],
        )
        entry = AuditLogEntry(event=event, formatted_message="Approve")
        await sink.write(entry)

        # Verify
        approve_events = sink.get_entries(phase=AuditPhase.APPROVE)
        assert len(approve_events) == 1
        assert approve_events[0].event.approved is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
